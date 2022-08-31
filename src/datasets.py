# %%
import torch
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DIR, VALID_DIR, CLASSES, BATCH_SIZE, RESIZE_TO
from utils import collate_fn, get_train_transform, get_valid_transform
import glob as glob
import cv2
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from xml.etree import ElementTree as et

# %%
class CustomImageDatasets(Dataset):
    def __init__(self, data_dir, classes, width, height, transforms = None):
        self.data_dir = data_dir
        self.classes = classes
        self.transforms = transforms
        self.width = width
        self.height = height

        self.image_dir_list = glob.glob(f"{self.data_dir}/*.jpg")
        self.all_images = [image_dir.split('/')[-1] for image_dir in self.image_dir_list]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # get image
        image_name = self.all_images[idx]
        image_dir = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        # TODO: check why this is needed
        image_resized = image_resized.transpose((2, 0, 1))
        image_resized = image/255.0

        # get label
        annot_name = image_name[:-4] + ".xml"
        annot_dir = os.path.join(self.data_dir, annot_name)

        tree = et.parse(annot_dir)
        root = tree.getroot()

        boxes = []
        labels = []
        
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        for obj in root.findall("object"):
            labels.append(self.classes.index(obj.find("name").text)) 

            xmin = int(obj.find("bndbox").find("xmin").text)
            xmax = int(obj.find("bndbox").find("xmax").text)
            ymin = int(obj.find("bndbox").find("ymin").text)
            ymax = int(obj.find("bndbox").find("ymax").text)

            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            # TODO: check why this is needed

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype = torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target["boxes"],
                                     labels = labels)
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

    def visualize_sample(self, idx):
        sample_image, sample_target = self.__getitem__(idx)
        
        fig, ax = plt.subplots()
        ax.imshow(sample_image.transpose((1, 2, 0)))

        for i in range(len(sample_target["boxes"])):
            box = sample_target["boxes"][i]
            label = CLASSES[sample_target["labels"][i]]

            xy = (int(box[0]), int(box[1]))
            height = int(box[3] - box[1])
            width = int(box[2] - box[0])
            rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
            ax.text(int(box[0]), int(box[1] - 5), label, fontsize = 10, color ="red")
            ax.add_patch(rect)
            

        plt.show()

# %%
train_dataset = CustomImageDatasets(TRAIN_DIR, CLASSES, RESIZE_TO, RESIZE_TO, get_train_transform())
valid_dataset = CustomImageDatasets(VALID_DIR, CLASSES, RESIZE_TO, RESIZE_TO, get_valid_transform())

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 0,
    collate_fn = collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 0,
    collate_fn = collate_fn
)

# %%
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")
