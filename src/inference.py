# %%
import os, sys

os.chdir('/root/custom-object-detection/src')

# %%
import torch

from config import TEST_DIR, NUM_CLASSES, CLASSES
from model import create_model
import torch
import glob as glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
detection_threshold = 0.8

# %%
model = create_model(NUM_CLASSES).to(device)
model.load_state_dict(torch.load("../outputs/model_100_final.pth", map_location = device))
model.eval()

# %%
test_images = glob.glob(f"{TEST_DIR}/*")
print(f"Test instances: {len(test_images)}")

# %%
print("Testing......\n")
for i in range(len(test_images)):
    image_name = test_images[i].split("/")[-1].split(".")[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype = torch.float, device = device)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image)
    
    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

    if (len(outputs[0]["boxes"]) != 0):
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()
        pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        
        fig, ax = plt.subplots()
        ax.imshow(orig_image)

        for j, box in enumerate(boxes):
            xy = (int(box[0]), int(box[1]))
            height = int(box[3] - box[1])
            width = int(box[2] - box[0])
            rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
            ax.text(int(box[0]), int(box[1] - 5), pred_classes[j], fontsize = 10, color ="red")
            ax.add_patch(rect)
        
        plt.savefig(f"../data/test_prediction/{image_name}.jpg")
        plt.close()
    
    print(f"Image {i + 1} done...")
    print("-"*50)
print("TEST PREDICTIon COMPLETE")
# %%
os.getcwd()
# %%
