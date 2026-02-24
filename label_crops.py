import sys
import math
from sklearn.metrics import confusion_matrix
import seaborn as sn
from datetime import datetime
import numpy as np
import random
from math import floor, ceil
from collections import defaultdict
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torchvision.transforms.v2 as transforms  # composable transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.io import decode_image
from torch.utils.data import random_split, Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def extract_info(filename):
    w, h, npb, npa = filename.split("_")[-4:]
    w = int(w)
    h = int(h)
    npb = int(npb)
    npa = int(npa.split(".")[0])  # Remove file extension
    return w, h, npb, npa

# Function to calculate mean and standard deviation of width and height
def calculate_mean_std_npb(directory):
    npb_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(tuple(IMG_EXTENSIONS)):  # Ensure it is an image file
                _, _, npb, _ = extract_info(filename)
                npb_list.append(npb)
    mean_npb = np.mean(npb_list)
    std_npb = np.std(npb_list)
    return mean_npb, std_npb

class SizeResNet50(ResNet):
    def _forward_impl(self, input):
        x = input["img"]
        npb = input["npb"]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        npb = npb.reshape((x.shape[0], 1))
        x = torch.cat((x, npb), 1)
        x = self.fc(x)

        return x

class OrigResNet50(ResNet):
    def _forward_impl(self, input):
        x = input["img"]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def my_resnet(
        block,
        layers,
        weights,
        progress,
        original=True,
        **kwargs,
) -> SizeResNet50:
    if original:
        model = OrigResNet50(block, layers, **kwargs)
    else:
        model = SizeResNet50(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def my_resnet50(*, original=True, weights=None, progress=True, **kwargs) -> SizeResNet50:
    weights = ResNet50_Weights.verify(weights)
    return my_resnet(Bottleneck, [3, 4, 6, 3], weights, progress, original=original, **kwargs)

class ImageFolderWithPaths(ImageFolder):
    def __init__(self, mean_npb, std_npb,
                root: str,
                transform=None,
                target_transform=None,
                is_valid_file=None,
                ):
            super().__init__(
                root,
                transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file,
            )
            self.mean_npb = mean_npb
            self.std_npb = std_npb

    def __getitem__(self, index): # obj[index] = {"npb", "path"}[Image, label]
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        npb_norm = None
        if self.mean_npb is not None:
            _, _, npb, _ = extract_info(os.path.basename(path))
            npb = int(npb)

            npb_norm = (npb - self.mean_npb) / self.std_npb
            npb_norm = torch.tensor(npb_norm, dtype=torch.float32)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {"npb": npb_norm, "path": path, "img": sample}, label


class EarlyStopping:
    def __init__(self, patience=1, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class SegmentClassifier():  # took out nn.Module inheritance bc of "cannot assign module before Module.__init__() call" error

    def __init__(self, id, data_dir, num_classes, device, mean_npb = None, std_npb = None, optim=1, Transform=None, sample=True, loss_weights=True,
                 batch_size=8, num_workers=0, lr=1e-4, stop_early=True, freeze_backbone=True):
        #############################################################################################################
        # data_dir: directory with images in subfolders, subfolders name are categories
        # num_classes: how many categories
        # Transform: data augmentations
        # sample: if the dataset is imbalanced, set to true and RandomWeightedSampler will be used
        # loss_weights: if the dataset is imbalanced, set to true and weight parameter will be passed to loss function
        # batch_size = number of samples used in 1 forward + backward pass thru network
        # num_workers = numer of processes putting data into RAM for processing
        # lr = learning rate
        # stop_early = whether to stop fitting once metric stops improving
        # freeze_backbone: if using pretrained architecture freeze all but the classification layer
        ###############################################################################################################
        self.val_loss = None
        self.id = id
        self.data_dir = data_dir  # 'data'
        self.num_classes = num_classes  # 4
        self.device = device
        self.mean_npb = mean_npb
        self.std_npb = std_npb
        self.optim = optim
        self.sample = sample
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        # self.weight_decay = weight_decay
        self.stop_early = stop_early
        self.freeze_backbone = freeze_backbone
        self.Transform = Transform
        self.val_predictions = []
        self.val_targets = []
        self.prev_val_acc = 0
        self.train_classes = None

    def load_data(self):
        '''
        # separate function for applying transforms
        # CSV of image filenames, randomly split based on class dist
        '''
        train_set = ImageFolderWithPaths(root = os.path.join(self.data_dir, "train"), mean_npb = self.mean_npb, std_npb = self.std_npb,
                                         transform=self.Transform)
        val_set = ImageFolderWithPaths(root = os.path.join(self.data_dir, "val"), mean_npb = self.mean_npb, std_npb = self.std_npb,
                                       transform=transforms.Compose([
                                           transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                                           transforms.Resize(size=(224, 224), antialias=True),
                                           transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input,
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])
                                       )
        self.train_classes = [i[1] for i in train_set]

        # because dataset is imbalanced:
        if self.sample:
            class_count = Counter(self.train_classes)  # how many of each class?
            # class_count[train_set.class_to_idx["Other"]] = round(class_count[train_set.class_to_idx["Other"]]/3)
            # divide the total # of images by # of each class
            # class_weights = 1./torch.tensor(class_count, dtype=torch.float)
            class_weights = torch.Tensor(
                [len(self.train_classes) / c for c in pd.Series(class_count).sort_index().values])

            sample_weights = [0] * len(train_set)
            for idx, (obj, label) in enumerate(train_set):
                class_weight = class_weights[label]
                sample_weights[idx] = class_weight

            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers,
                                      sampler=sampler)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        else:
            train_loader = DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=2, num_workers=self.num_workers)

        return train_loader, val_loader

    def load_model(self, arch='resnet', mode="new"):
        self.model = my_resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, original = (self.mean_npb is None))
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        #    for param in self.model.fc.parameters():
        #     param.requires_grad = True
        self.model.fc = nn.Sequential(nn.Linear(in_features=self.model.fc.in_features+(self.mean_npb is None), out_features=256),
                                      nn.ReLU(inplace=True),
                                      #nn.Dropout(0.4),
                                      nn.Linear(in_features=256, out_features=self.num_classes))

        for param in self.model.fc.parameters():
            param.requires_grad = True

        for name, layer in self.model.named_children():
            if name in ['conv1', 'bn1', 'layer1', 'layer2']:
                for param in layer.parameters():
                    param.requires_grad = False

        # for param in self.model.layer4.parameters():
        #     param.requires_grad = True

        with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training\\run_notes.csv", 'a') as rn:
            rn.write('ID, Epoch, Training_loss, validation_loss, validation_accuracy' + '\n')

        with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training\\val_notes.csv", 'a') as vn:
            vn.write('ID, Epoch, Class, Prediction, Filename' + '\n')

        self.model = self.model.to(self.device)

        if self.optim == 1:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == 2:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        if self.loss_weights:
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor(
                [len(self.train_classes) / c for c in pd.Series(class_count).sort_index().values])
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        if mode == "existing":
            self.model = torch.load(
                "C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training\\SegmentClassifier_2025_03_27.pt",
                weights_only=False)

    def fit_one_epoch(self, train_loader, epoch, num_epochs, start_timestamp):
        epoch_train_losses = list()
        epoch_train_accs = list()
        train_targets = [0] * self.num_classes
        self.model.train()
#        for idx, (inputs, labels) in enumerate(tqdm(train_loader, position=0, leave=True)):
        for idx, (inputs, labels) in enumerate(tqdm(train_loader, position=0, leave=True)):
            inputs = {key: value.to(self.device) if hasattr(value, 'to') else value for key, value in inputs.items()}
            labels = labels.to(self.device)
            # images = obj["img"].to(self.device)
            # targets = label.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            epoch_train_losses.append(loss.item())

            predictions = torch.argmax(logits, dim=1)
            num_correct = sum(predictions.eq(labels))
            running_train_acc = float(num_correct) / float(inputs["img"].shape[0])
            epoch_train_accs.append(running_train_acc)
            for label in labels:  # keep track of how many of each class the model is being trained on
                train_targets[label.cpu().numpy()] += 1

        train_loss = torch.tensor(epoch_train_losses).mean()
        train_acc = torch.tensor(epoch_train_accs).mean()

        time_diff = datetime.today() - start_timestamp

        print(f'Epoch {epoch} /{num_epochs - 1}, {time_diff}')
        print(f'Training loss: {train_loss:.2f}')
        print(f'Class distribution: {train_targets}')
        print(f'Learning rate: {self.lr}')
        with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training\\run_notes.csv", 'a') as rn:
            rn.write(f'{self.id},{epoch},{train_loss},')
        return {
            'train_loss': train_loss,
            'train_acc': train_acc
        }

    def val_one_epoch(self, val_loader, epoch):
        epoch_val_losses = list()
        epoch_val_accs = list()
        val_targets = [0] * self.num_classes
        self.model.eval()
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs = {key: value.to(self.device) if hasattr(value, 'to') else value for key, value in inputs.items()}
                labels = labels.to(self.device)
                # images = obj["img"].to(self.device)
                # targets = label.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                epoch_val_losses.append(loss.item())

                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(labels))
                running_val_acc = float(num_correct) / float(inputs["img"].shape[0])

                epoch_val_accs.append(running_val_acc)
                for target in labels:  # keep track of how many of each class the model is being trained on
                    val_targets[target.cpu().numpy()] += 1
                correct_per_class = [0] * self.num_classes
                with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training\\val_notes.csv", 'a') as vn:
                    for i in range(len(inputs["path"])):
                        vn.write(f'{self.id},{epoch},{labels[i]},{predictions[i]},{inputs["path"][i]}\n')
                        self.val_targets.append(labels[i].cpu().numpy())
                        self.val_predictions.append(predictions[i].cpu().numpy())
                        # print how many are correct in each category
                        correct_per_class[labels[i].item()] += (labels[i] == predictions[i]).item()

            self.val_loss = torch.tensor(epoch_val_losses).mean()
            val_acc = torch.tensor(epoch_val_accs).mean()  # average acc per batch

            print(f'validation loss: {self.val_loss:.2f}')
            print(f'validation accuracy: {val_acc:.2f}')
            print(f'Class distribution: {val_targets}')
            print(f'Correct: {correct_per_class}')
            with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training\\run_notes.csv", 'a') as rn:
                rn.write(f'{self.val_loss},{val_acc},\n')
            return {
                'val_loss': self.val_loss,
                'val_acc': val_acc
            }

    def fit(self, train_loader, val_loader, num_epochs=10, unfreeze_after=5, checkpoint_dir='checkpoint.pt'):
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        if self.stop_early:
            early_stopping = EarlyStopping(patience=50, path=checkpoint_dir)
        start_timestamp = datetime.today()
        for epoch in range(num_epochs):
            if self.freeze_backbone:
                if epoch == unfreeze_after:
                    for param in self.model.parameters():
                        param.requires_grad = True
            train_metrics = self.fit_one_epoch(train_loader, epoch, num_epochs, start_timestamp)
            train_losses.append(train_metrics["train_loss"])
            train_accuracies.append(train_metrics["train_acc"])
            self.val_targets.clear()
            self.val_predictions.clear()
            val_metrics = self.val_one_epoch(val_loader, epoch)
            val_losses.append(val_metrics["val_loss"])
            val_accuracies.append(val_metrics["val_acc"])

            if self.stop_early:
                early_stopping(self.val_loss, self.model)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    print(f'Best validation loss: {early_stopping.best_score}')
        return {
            'train_losses': train_losses,
            'train_acc': train_accuracies,
            'val_losses': val_losses,
            'val_acc': val_accuracies,
        }

    def compute_confusion_matrix(preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        return confusion_matrix(y, rounded_preds)

    def head_dict(d, limit):
        for element in d:
            print(d[element][0:limit])

    def dim_dict(d):
        for element in d:
            print(element, ": ", len(d[element]))