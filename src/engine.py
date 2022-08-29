# %%
import os, sys

os.chdir("/root/custom-object-detection/src")
os.getcwd()

# %%
from datasets import train_loader, valid_loader
from config import DEVICE, NUM_EPOCHS, NUM_CLASSES, OUT_DIR, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH
from utils import Averager
from model import create_model

import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time


# %%
plt.style.use("ggplot")

# %%
def train(train_loader, model):
    print("Training......")

    global train_itr
    global train_loss_list

    prog_bar = tqdm(train_loader, total = len(train_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = [torch.from_numpy(image).to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # losses for classifer, box regression, objectness, brpn box regression
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_value = loss.item()
        train_loss_list.append(loss_value)
        train_loss_history.send(loss_value)

        loss.backward()
        optimizer.step()

        train_itr += 1

        prog_bar.set_description(desc = f"Loss: {loss.item()}")

    return train_loss_list

# %%
def validate(valid_loader, model):
    print("Validating......")

    global valid_itr
    global valid_loss_list

    prog_bar = tqdm(valid_loader, total = len(valid_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = [torch.from_numpy(image).to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # losses for classifer, box regression, objectness, brpn box regression
        with torch.no_grad():
            loss_dict = model(images, targets)
        
        loss = sum(loss for loss in loss_dict.values())
        loss_value = loss.item()
        valid_loss_list.append(loss_value)
        valid_loss_history.send(loss_value)

        valid_itr += 1

        prog_bar.set_description(desc = f"Loss: {loss.item()}")

    return valid_loss_list
# %%
model = create_model(num_classes = NUM_CLASSES)
model = model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

train_itr = 1
valid_itr = 1
train_loss_list = []
valid_loss_list = []
train_loss_history = Averager()
valid_loss_history = Averager()

MODEL_NAME = "Fast-RCNN"
# %%
for epoch in range(NUM_EPOCHS):
    print(f"======== EPOCH {epoch + 1} of {NUM_EPOCHS} ========")

    fig, axs = plt.subplots(2, 1)
    
    train_loss_history.reset()
    valid_loss_history.reset()

    start = time.time()
    train_loss_list = train(train_loader, model)
    valid_loss_list = validate(valid_loader, model)

    print(f"Epoch #{epoch} train loss: {train_loss_history.value:.3f}")   
    print(f"Epoch #{epoch} validation loss: {valid_loss_history.value:.3f}") 
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
    if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
        torch.save(model.state_dict(), f"{OUT_DIR}/model_{epoch + 1}.pth")
        print("SAVING MODEL COMPLETE...\n")

    if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
        axs[0].plot(train_loss_list, color = "blue")
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Train loss")

        axs[1].plot(valid_loss_list, color = "red")  
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Valid loss")

        plt.gcf().set_size_inches(15, 12)
        plt.savefig(f"{OUT_DIR}/train_val_loss_history_{epoch + 1}.png")
        print("SAVING PLOT COMPLETE...\n")


    if (epoch + 1) == NUM_EPOCHS:
        torch.save(model.state_dict(), f"{OUT_DIR}/model_{epoch + 1}_final.pth")
        
        axs[0].plot(train_loss_list, color = "blue")
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Train loss")

        axs[1].plot(valid_loss_list, color = "red")  
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Valid loss")

        plt.gcf().set_size_inches(15, 12)
        plt.savefig(f"{OUT_DIR}/train_val_loss_history_{epoch + 1}_final.png")

        print("SAVING FINAL MODEL AND PLOT COMPLETE...\n")

# %%
