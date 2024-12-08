import os
from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent))  # Add the project root to sys.path
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.oct_dataset import OCTDataset
from utils.models import UnetNoPretraining
from utils.lossfunctions import DiceLoss, DiceBCELoss 
from utils.data_augmentation import DataAugmentTransform
from tqdm import tqdm
import torch.optim as optim
import segmentation_models_pytorch as smp

root_dir = "/data/data_gentuity"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare datasets for the fold
transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# Initialize model with the hyperparameters from the config
net = smp.Unet(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

model_state, optimizer_state = torch.load("/data/best_checkpoints/first_iteration/unet_unfrozen_bs=6_dicebce.pt", weights_only=True)
net.load_state_dict(model_state)

# Freeze the encoder by setting requires_grad=False
for param in net.encoder.parameters():
    param.requires_grad = False

# Define loss and optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# Define the loss function
criterion = DiceBCELoss()

####
# Define splits
with open(os.path.join(root_dir, "metadata.csv"), "r") as f:
    metadata_df = pd.read_csv(f)
    skf = StratifiedKFold(n_splits=5)
    splits = list(skf.split(metadata_df, metadata_df["unique_id"]))

#Use only the first split
train_indices, val_indices = splits[1]

# Create the training and validation datasets
train_dataset = OCTDataset(root_dir, indices=train_indices, train=True, is_gentuity=True, transform=transform)

# Apply augmentation to the training dataset
sample_size = len(train_dataset)//3

# Instantiate the combined transform
data_augment_transform = DataAugmentTransform()

# Randomly sample a subset of the training dataset
aug_indices = np.random.choice(train_indices, sample_size, replace=False)

# Create a new dataset for augmentation
aug_dataset = OCTDataset(root_dir, indices=aug_indices, train=True, is_gentuity=True, transform=data_augment_transform, for_augmentation=True)

# Combine the original and augmented datasets
train_dataset = ConcatDataset([train_dataset, aug_dataset])

val_dataset = OCTDataset(root_dir, indices=val_indices, train=True, is_gentuity=True, transform=transform)

trainloader = DataLoader(train_dataset, batch_size=6, shuffle=False, num_workers=8, drop_last=True)
valloader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=8, drop_last=True)

# Move the model to the appropriate device
net.to(device)

epochs = 10

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    correct_train_predictions = 0
    total_train_pixels = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for images, masks, _, _ in progress_bar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item())

        # Calculate training accuracy (pixel-wise accuracy)
        pred = (outputs > 0.5).float()  # Assuming binary classification (adjust for multi-class)
        correct_train_predictions += torch.sum(pred == masks).item()
        total_train_pixels += masks.numel()

        progress_bar.set_postfix(loss=loss.item())

    # Calculate training loss and accuracy for the epoch
    train_loss = running_loss / len(trainloader.dataset)
    train_accuracy = correct_train_predictions / total_train_pixels
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # Validation phase
    net.eval()
    val_loss = 0.0
    correct_val_predictions = 0
    total_val_pixels = 0
    with torch.no_grad():  # No need to calculate gradients during validation
        for images, masks, _, _ in tqdm(valloader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = net(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            # Calculate validation accuracy (pixel-wise accuracy)
            pred = (outputs > 0.5).float()  # Assuming binary classification (adjust for multi-class)
            correct_val_predictions += torch.sum(pred == masks).item()
            total_val_pixels += masks.numel()

    # Calculate validation loss and accuracy
    val_loss = val_loss / len(valloader.dataset)
    val_accuracy = correct_val_predictions / total_val_pixels
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")