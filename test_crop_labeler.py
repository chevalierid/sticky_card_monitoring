import torch, os, sys, glob

os.chdir('C:/Users/ALANalysis/flat-bug/src/A_rubi_positive_training')
sys.path.insert(0, os.getcwd())
from PIL import Image
#from label_crops import SegmentClassifier
import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms  # composable transforms
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
import matplotlib.pyplot as plt
import shutil
import argparse
import gc
gc.collect()

def cli_args():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument("-d", "--model_type", type=str, dest="model_type", required=True,
                            help="Whether loading in full model or dictionary")
    args = args_parse.parse_args()
    return vars(args)

class ClassifierTest(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.images = [f for f in os.listdir(self.dir) if (os.path.isfile(os.path.join(self.dir,f)) & f.lower().endswith(('.png', '.jpg', '.jpeg')))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #print(os.path.join(self.dir, self.images[idx]))
        img = Image.open(os.path.join(self.dir, self.images[index]))
        return self.transform(img), self.images[index]

def classify_segments(model_type):
    #torch.cuda.empty_cache()
    #PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #data_dir = "./data/validation_nofolders"
    #data_dir = "C:\\Users\\ALANalysis\\flat-bug\\src\\2024_chandra_moffat_crops"
    data_dir = "C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\data\\test"

    # for png_file in glob.glob(data_dir + "\\**\\crops\\*.png"):
    #     #print(png_file)
    #     shutil.move(png_file, data_dir)
    pretrained = torch.load("C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_positive_training\\SegmentClassifier_4.27_03-26-18-36.pt", weights_only = False)
    pretrained.eval()

    # check whether val_set results match last epoch or best epoch
    val_set = ClassifierTest(data_dir, transform=transforms.Compose([
        transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        transforms.Lambda(lambda x: x[:3])
        # remove alpha channel since the model's dataset is built with ImageFolder which is RGB
    ]))

    #val_loader = DataLoader(val_set, batch_size=len(val_set))
    val_loader = DataLoader(val_set, batch_size=16)

    sub = pd.DataFrame(columns=['category', 'id'])
    id_list = []
    pred_list = []

    pretrained = pretrained.to(device)

    with torch.no_grad():
        for (image, image_id) in val_loader:
            image = image.to(device)

            logits = pretrained(image)
            predicted = list(torch.argmax(logits, 1).cpu().numpy())

            for id in image_id:
                id_list.append(id)

            for prediction in predicted:
                pred_list.append(prediction.tolist())

    sub['category'] = pred_list
    sub['id'] = id_list

    #mapping = {0: 'Other', 1: 'SWD_male', 2: 'SWD_parasitoid', 3: 'Weevil'}
    mapping = {0: 'Debris', 1: 'Arthropod', 2: 'SWD_male', 3: 'SWD_parasitoid', 4: 'Weevil'}

    # make subfolders for all classes
    for key,subfolder in mapping.items():
        sub_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(sub_path):
            os.mkdir(os.path.join(sub_path))

    sub['category'] = sub['category'].map(mapping)
    sub = sub.sort_values(by='id')
    print(sub)

    ground_truths = []
    for name in sub['id']:
        ground_truths.append(re.sub(r"_(\d+).png", '', name))
        # move
        shutil.copyfile(os.path.join(data_dir, name), os.path.join(data_dir, sub.loc[sub['id'] == name, 'category'].item(), name))
    # val_targets_labels = []
    # val_preds_labels = []
    # idx2class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}

    # for (target, pred) in zip(pretrained.val_targets, classifier.val_predictions):
    #     val_targets_labels.append(idx2class[target.max()])
    #     val_preds_labels.append(idx2class[pred.max()])

    # cm = confusion_matrix(val_targets_labels, val_preds_labels)
    cm = confusion_matrix(sub['category'], ground_truths)
    # ConfusionMatrixDisplay(cm, display_labels = list(val_loader.dataset.class_to_idx.keys())).plot()
    ConfusionMatrixDisplay(cm, display_labels=mapping.values()).plot()
    plt.show()

def main():
    classify_segments(**cli_args())


if __name__ == '__main__':
    main()
