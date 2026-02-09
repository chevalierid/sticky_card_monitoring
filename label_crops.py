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
from torchvision.transforms import RandomRotation
from torchvision.datasets import ImageFolder
from torchvision.io import decode_image
from torch.utils.data import random_split, Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class CustomDataset(ImageFolder):
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        sample = self.loader(path)
        return path, sample, label


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

    def __init__(self, id, data_dir, num_classes, device, optim=1, Transform=None, sample=True, loss_weights=True,
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
        train_set = ImageFolder(os.path.join(self.data_dir, "train"), transform=self.Transform)
        val_set = ImageFolderWithPaths(os.path.join(self.data_dir, "val"), transform=transforms.Compose([
            transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
                                       )

        # train_files = ImageFolder(root = self.data_dir + f"/training", transform = self.Transform)
        # val_files = ImageFolderWithPaths(root = self.data_dir + f"/validation", transform = transforms.Compose([
        #     transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        #     transforms.Resize(size = (224, 224), antialias=True),
        #     transforms.ToDtype(torch.float32, scale=True)  # Normalize expects float input
        #     ])
        # )

        # full_set = ImageFolder(root = self.data_dir + f"/all")
        # train_set, val_set = torch.utils.data.random_split(full_set, [0.8, 0.2])
        # # val_set = ImageFolderWithPaths(root = self.data_dir + f"/validation", transform = transforms.Compose([
        # #     transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        # #     transforms.Resize(size = (224, 224), antialias=True),
        # #     transforms.ToDtype(torch.float32, scale=True)  # Normalize expects float input
        # #     ])
        # )

        self.train_classes = [label for _, label in train_set]

        # because dataset is imbalanced:
        if self.sample:
            class_count = Counter(self.train_classes)  # how many of each class?
            # class_count[train_set.class_to_idx["Other"]] = round(class_count[train_set.class_to_idx["Other"]]/3)
            # divide the total # of images by # of each class
            # class_weights = 1./torch.tensor(class_count, dtype=torch.float)
            class_weights = torch.Tensor(
                [len(self.train_classes) / c for c in pd.Series(class_count).sort_index().values])

            sample_weights = [0] * len(train_set)
            for idx, (image, label) in enumerate(train_set):
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
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        #    for param in self.model.fc.parameters():
        #     param.requires_grad = True
        self.model.fc = nn.Sequential(nn.Linear(in_features=self.model.fc.in_features, out_features=256),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.4),
                                      nn.Linear(in_features=256, out_features=self.num_classes))
        # nn.Dropout(0.5))               # Dropout layer with 50% probability

        for param in self.model.fc.parameters():
            param.requires_grad = True

        for name, layer in self.model.named_children():
            if name in ['conv1', 'bn1', 'layer1', 'layer2']:
                for param in layer.parameters():
                    param.requires_grad = False

        # for param in self.model.layer4.parameters():
        #     param.requires_grad = True

        with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\run_notes.csv", 'a') as rn:
            rn.write('ID, Epoch, Training_loss, validation_loss, validation_accuracy' + '\n')

        with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\val_notes.csv", 'a') as vn:
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
                "C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\SegmentClassifier_2025_03_27.pt",
                weights_only=False)

    def fit_one_epoch(self, train_loader, epoch, num_epochs, start_timestamp):
        epoch_train_losses = list()
        epoch_train_accs = list()
        train_targets = [0, 0, 0, 0, 0]
        self.model.train()
        for i, (images, targets) in enumerate(tqdm(train_loader, position=0, leave=True)):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            epoch_train_losses.append(loss.item())

            predictions = torch.argmax(logits, dim=1)
            num_correct = sum(predictions.eq(targets))
            running_train_acc = float(num_correct) / float(images.shape[0])
            epoch_train_accs.append(running_train_acc)
            for target in targets:  # keep track of how many of each class the model is being trained on
                train_targets[target.cpu().numpy()] += 1

        train_loss = torch.tensor(epoch_train_losses).mean()
        train_acc = torch.tensor(epoch_train_accs).mean()

        time_diff = datetime.today() - start_timestamp

        print(f'Epoch {epoch} /{num_epochs - 1}, {time_diff}')
        print(f'Training loss: {train_loss:.2f}')
        print(f'Class distribution: {train_targets}')
        print(f'Learning rate: {self.lr}')
        with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\run_notes.csv", 'a') as rn:
            rn.write(f'{self.id},{epoch},{train_loss},')
        return {
            'train_loss': train_loss,
            'train_acc': train_acc
        }

    def val_one_epoch(self, val_loader, epoch):
        epoch_val_losses = list()
        epoch_val_accs = list()
        val_targets = [0, 0, 0, 0, 0]
        self.model.eval()
        with torch.no_grad():
            for (images, targets, paths) in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                epoch_val_losses.append(loss.item())

                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                running_val_acc = float(num_correct) / float(images.shape[0])

                epoch_val_accs.append(running_val_acc)
                for target in targets:  # keep track of how many of each class the model is being trained on
                    val_targets[target.cpu().numpy()] += 1
                correct_per_class = [0] * self.num_classes
                with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\val_notes.csv", 'a') as vn:
                    for i in range(len(paths)):
                        vn.write(f'{self.id},{epoch},{targets[i]},{predictions[i]},{paths[i]}\n')
                        self.val_targets.append(targets[i].cpu().numpy())
                        self.val_predictions.append(predictions[i].cpu().numpy())
                        # print how many are correct in each category
                        correct_per_class[targets[i].item()] += (targets[i] == predictions[i]).item()

            self.val_loss = torch.tensor(epoch_val_losses).mean()
            val_acc = torch.tensor(epoch_val_accs).mean()  # average acc per batch

            print(f'validation loss: {self.val_loss:.2f}')
            print(f'validation accuracy: {val_acc:.2f}')
            print(f'Class distribution: {val_targets}')
            print(f'Correct: {correct_per_class}')
            with open("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\run_notes.csv", 'a') as rn:
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