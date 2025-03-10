import sys
import math
import pandas as pd
from collections import Counter
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as transforms              # composable transforms
from torchvision.transforms import RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    class EarlyStopping:
        def __init__(self, patience=1, delta=0, path='checkpoint.pt'):
            self.patience = patience
            self.delta = delta
            self.path= path
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def __call__(self, val_loss, model):
            if self.best_score is None:
                self.best_score = val_loss
                self.save_checkpoint(model)
            elif val_loss > self.best_score:
                self.counter +=1
                if self.counter >= self.patience:
                    self.early_stop = True 
            else:
                self.best_score = val_loss
                self.save_checkpoint(model)
                self.counter = 0      

        def save_checkpoint(self, model):
            torch.save(model.state_dict(), self.path)





    class SegmentClassifier():
    
        def __init__(self, data_dir, num_classes, device, optim = 1, Transform = None, sample = True, loss_weights = True, batch_size = 8, num_workers = 0, lr = 1e-4, stop_early = True, freeze_backbone = True):
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
            self.data_dir = data_dir # 'data'
            self.num_classes = num_classes # 4
            self.device = device
            self.optim = optim
            self.sample = sample
            self.loss_weights = loss_weights
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.lr = lr
            #self.weight_decay = weight_decay
            self.stop_early = stop_early
            self.freeze_backbone = freeze_backbone
            self.Transform = Transform
            
        def load_data(self):
            dataset = ImageFolder(root = self.data_dir, transform = self.Transform)

            train_set, val_set = random_split(dataset, [math.floor(len(dataset)*0.8), math.ceil(len(dataset)*0.2)])

            #train_set = train_set()

            self.train_classes = [label for _, label in train_set]

            # because dataset is imbalanced:
            if self.sample:
                class_count = Counter(self.train_classes)
                class_weights = torch.Tensor([len(self.train_classes)/c for c in pd.Series(class_count).sort_index().values])

                sample_weights = [0] * len(train_set)
                for idx, (image, label) in enumerate(train_set):
                    class_weight = class_weights[label]
                    sample_weights[idx] = class_weight
                # 80/20 split:
                #     80 weevils
                #         16 in validation
                #     100 SWD parasitoid
                #         16
                #     200 SWD male
                #         16
                #     30 000 OTHER
                #         16

                sampler = WeightedRandomSampler(weights = sample_weights,
                                                num_samples = len(train_set), replacement = True)
                train_loader = DataLoader(train_set, batch_size = self.batch_size, num_workers = self.num_workers, sampler = sampler)
                val_loader = DataLoader(val_set, batch_size = self.batch_size, num_workers = self.num_workers, sampler = sampler)
            else:
                train_loader = DataLoader(train_set, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)
                val_loader = DataLoader(val_set, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)
            
            return train_loader, val_loader



        def load_model(self, arch = 'resnet'):
            self.model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            #    for param in self.model.fc.parameters():
            #     param.requires_grad = True
            self.model.fc = nn.Sequential(
                nn.Linear(in_features = self.model.fc.in_features, out_features = self.num_classes))
                #nn.ReLU(inplace=True),
                #nn.Dropout(0.5))               # Dropout layer with 50% probability




            self.model = self.model.to(self.device)

            if self.optim == 1:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
            elif self.optim == 2:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr)

            if self.loss_weights:
                class_count = Counter(self.train_classes)
                class_weights = torch.Tensor([len(self.train_classes)/c for c in pd.Series(class_count).sort_index().values])
                class_weights = class_weights.to(self.device)
                self.criterion = nn.CrossEntropyLoss(class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()




        def fit_one_epoch(self, train_loader, epoch, num_epochs):
            train_losses = list()
            train_acc = list()
            self.model.train()
            for i, (images, targets) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_losses.append(loss.item())

                predictions = torch.argmax(logits, dim = 1)
                num_correct = sum(predictions.eq(targets))
                running_train_acc = float(num_correct) / float(images.shape[0])
                train_acc.append(running_train_acc)

            train_loss = torch.tensor(train_losses).mean()
            print(f'Epoch {epoch} /{num_epochs - 1}')
            print(f'Training loss: {train_loss:.2f}')

        



        def val_one_epoch(self, val_loader):
            val_losses = list()
            val_accs = list()
            self.model.eval()
            with torch.no_grad():
                for (images, targets) in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                    val_losses.append(loss.item())

                    predictions = torch.argmax(logits, dim = 1)
                    num_correct = sum(predictions.eq(targets))
                    running_val_acc = float(num_correct) / float(images.shape[0])

                    val_accs.append(running_val_acc)
                self.val_loss = torch.tensor(val_losses).mean()
                val_acc = torch.tensor(val_accs).mean() # average acc per batch

                print(f'Validation loss: {self.val_loss:.2f}')
                print(f'Validation accuracy: {val_acc:.2f}')



            
        def fit(self, train_loader, val_loader, num_epochs = 10, unfreeze_after = 5, checkpoint_dir = 'checkpoint.pt'):
            if self.stop_early:
                early_stopping = EarlyStopping(patience = 5, path = checkpoint_dir)
                
            for epoch in range(num_epochs):
                    if self.freeze_backbone:
                        if epoch == unfreeze_after:
                            for param in self.model.parameters():
                                param.requires_grad = True
                    self.fit_one_epoch(train_loader, epoch, num_epochs)
                    self.val_one_epoch(val_loader)
                    if self.stop_early:
                        early_stopping(self.val_loss, self.model)
                        if early_stopping.early_stop:
                            print('Early Stopping')
                            print(f'Best validation loss: {early_stopping.best_score}')


    Transform = transforms.Compose([
            transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            #transforms.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
            transforms.Resize(size = (224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True)  # Normalize expects float input
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data"

    classifier_0 = SegmentClassifier(data_dir = data_dir, num_classes = 4, device = device, optim = 1, lr = 1e-2, batch_size = 32, num_workers = 4, Transform = Transform, sample = True, loss_weights = True)
    train_loader, val_loader = classifier_0.load_data()
    classifier_0.load_model()
    classifier_0.fit(num_epochs = 20, unfreeze_after = 5, train_loader = train_loader, val_loader = val_loader)
    try:
        torch.save(classifier_0.model, "SegmentClassifier_0_M.pt")
    except:
        print("Could not save")
        
    classifier_1 = SegmentClassifier(data_dir = data_dir, num_classes = 4, device = device, optim = 2, lr = 1e-4, batch_size = 16, num_workers = 4, Transform = Transform, sample = True, loss_weights = True)
    train_loader, val_loader = classifier_1.load_data()
    classifier_1.load_model()
    classifier_1.fit(num_epochs = 20, unfreeze_after = 5, train_loader = train_loader, val_loader = val_loader)
    try:
        torch.save(classifier_1.model, "SegmentClassifier_1_M.pt")
    except:
        print("Could not save")

    classifier_2 = SegmentClassifier(data_dir = data_dir, num_classes = 4, device = device, optim = 2, lr = 1e-5, batch_size = 16, num_workers = 8, Transform = Transform, sample = True, loss_weights = True)
    train_loader, val_loader = classifier_2.load_data()
    classifier_2.load_model()
    classifier_2.fit(num_epochs = 20, unfreeze_after = 5, train_loader = train_loader, val_loader = val_loader)
    try:
        torch.save(classifier_2.model, "SegmentClassifier_2_M.pt")
    except:
        print("Could not save")

    classifier_3 = SegmentClassifier(data_dir = data_dir, num_classes = 4, device = device, optim = 2, lr = 1e-2, batch_size = 32, num_workers = 8, Transform = Transform, sample = True, loss_weights = True)
    train_loader, val_loader = classifier_3.load_data()
    classifier_3.load_model()
    classifier_3.fit(num_epochs = 20, unfreeze_after = 5, train_loader = train_loader, val_loader = val_loader)
    try:
        torch.save(classifier_3.model, "SegmentClassifier_3_M.pt")
    except:
        print("Could not save")
    

        

    

if __name__ == '__main__':
    main()