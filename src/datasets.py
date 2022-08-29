# %%
import torch
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DIR, VALID_DIR, CLASSES, BATCH_SIZE
from utils import collate_fn
import glob as glob
import cv2
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from xml.etree import ElementTree as et

# %%
class CustomImageDatasets(Dataset):
    def __init__(self, data_dir, classes, transforms = None):
        self.data_dir = data_dir
        self.classes = classes
        self.transforms = transforms

        self.image_dir_list = glob.glob(f"{self.data_dir}/*.jpg")
        self.all_images = [image_dir.split('/')[-1] for image_dir in self.image_dir_list]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # get image
        image_name = self.all_images[idx]
        image_dir = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image_resized = cv2.resize(image, (self.width, self.height))
        # TODO: check why this is needed
        image = image.transpose((2, 0, 1))
        image_resized = image/255.0

        # get label
        annot_name = image_name[:-4] + ".xml"
        annot_dir = os.path.join(self.data_dir, annot_name)

        tree = et.parse(annot_dir)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            labels.append(self.classes.index(obj.find("name").text)) 

            xmin = int(obj.find("bndbox").find("xmin").text)
            xmax = int(obj.find("bndbox").find("xmax").text)
            ymin = int(obj.find("bndbox").find("ymin").text)
            ymax = int(obj.find("bndbox").find("ymax").text)

            # xmin_final = (xmin/image_width)*self.width
            # xmax_final = (xmax/image_width)*self.width
            # ymin_final = (ymin/image_height)*self.height
            # yamx_final = (ymax/image_height)*self.height
            # TODO: check why this is needed

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype = torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels

        
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
# class MicrocontrollerDataset(Dataset):
#     def __init__(self, dir_path, width, height, classes, transforms=None):
#         self.transforms = transforms
#         self.dir_path = dir_path
#         self.height = height
#         self.width = width
#         self.classes = classes
        
#         # get all the image paths in sorted order
#         self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
#         self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
#         self.all_images = sorted(self.all_images)
#     def __getitem__(self, idx):
#         # capture the image name and the full image path
#         image_name = self.all_images[idx]
#         image_path = os.path.join(self.dir_path, image_name)
#         # read the image
#         image = cv2.imread(image_path)
#         # convert BGR to RGB color format
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#         image_resized = cv2.resize(image, (self.width, self.height))
#         image_resized /= 255.0
        
#         # capture the corresponding XML file for getting the annotations
#         annot_filename = image_name[:-4] + '.xml'
#         annot_file_path = os.path.join(self.dir_path, annot_filename)
        
#         boxes = []
#         labels = []
#         tree = et.parse(annot_file_path)
#         root = tree.getroot()
        
#         # get the height and width of the image
#         image_width = image.shape[1]
#         image_height = image.shape[0]
        
#         # box coordinates for xml files are extracted and corrected for image size given
#         for member in root.findall('object'):
#             # map the current object name to `classes` list to get...
#             # ... the label index and append to `labels` list
#             labels.append(self.classes.index(member.find('name').text))
            
#             # xmin = left corner x-coordinates
#             xmin = int(member.find('bndbox').find('xmin').text)
#             # xmax = right corner x-coordinates
#             xmax = int(member.find('bndbox').find('xmax').text)
#             # ymin = left corner y-coordinates
#             ymin = int(member.find('bndbox').find('ymin').text)
#             # ymax = right corner y-coordinates
#             ymax = int(member.find('bndbox').find('ymax').text)
            
#             # resize the bounding boxes according to the...
#             # ... desired `width`, `height`
#             xmin_final = (xmin/image_width)*self.width
#             xmax_final = (xmax/image_width)*self.width
#             ymin_final = (ymin/image_height)*self.height
#             yamx_final = (ymax/image_height)*self.height
            
#             boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        
#         # bounding box to tensor
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # area of the bounding boxes
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # no crowd instances
#         iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
#         # labels to tensor
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         # prepare the final `target` dictionary
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#         image_id = torch.tensor([idx])
#         target["image_id"] = image_id
#         # apply the image transforms
#         if self.transforms:
#             sample = self.transforms(image = image_resized,
#                                      bboxes = target['boxes'],
#                                      labels = labels)
#             image_resized = sample['image']
#             target['boxes'] = torch.Tensor(sample['bboxes'])
            
#         return image_resized, target
#     def __len__(self):
#         return len(self.all_images)
# %%
train_dataset = CustomImageDatasets(TRAIN_DIR, CLASSES)
valid_dataset = CustomImageDatasets(VALID_DIR, CLASSES)

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
# %%
