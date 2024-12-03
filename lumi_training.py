import os
import argparse
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.oct_dataset import OCTDataset
from utils.models import UnetNoPretraining
from utils.lossfunctions import DiceLoss, DiceBCELoss
from tqdm import tqdm
import torch.optim as optim
import segmentation_models_pytorch as smp 

# # Set up argument parser to accept root_dir
# parser = argparse.ArgumentParser()
# parser.add_argument('--root_dir', type=str, required=True, help="Path to the data directory")
# args = parser.parse_args()

# root_dir = args.root_dir
# print(f"Root directory: {root_dir}")

root_dir = "/data/data_terumo_smoke_test"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare datasets for the fold
transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# Initialize model with the hyperparameters from the config
net = smp.DeepLabV3Plus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation="sigmoid",           # output activation (sigmoid for binary segmentation)
)

# Freeze the encoder by setting requires_grad=False
for param in net.encoder.parameters():
    param.requires_grad = False

# Define loss and optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# Define the loss function
criterion = DiceBCELoss()

# Split the dataset into training and validation sets
dataset = OCTDataset(root_dir, transform=transform)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(len(train_dataset))

batch_size = 64

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Get a batch of training data
train_images, train_masks, _, _ = next(iter(trainloader))
val_images, val_masks, _, _ = next(iter(valloader))

# Move the images and masks to the appropriate device
train_images, train_masks = train_images.to(device), train_masks.to(device)
val_images, val_masks = val_images.to(device), val_masks.to(device)

# Plot training images and masks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axes[0, i].imshow(train_images[i].permute(1, 2, 0).cpu().numpy())
    axes[0, i].set_title(f"Train Image {i+1}")
    axes[0, i].axis('off')
    axes[1, i].imshow(train_masks[i].squeeze().cpu().numpy(), cmap='gray')
    axes[1, i].set_title(f"Train Mask {i+1}")
    axes[1, i].axis('off')
plt.show()

# Plot validation images and masks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    axes[0, i].imshow(val_images[i].permute(1, 2, 0).cpu().numpy())
    axes[0, i].set_title(f"Val Image {i+1}")
    axes[0, i].axis('off')
    axes[1, i].imshow(val_masks[i].squeeze().cpu().numpy(), cmap='gray')
    axes[1, i].set_title(f"Val Mask {i+1}")
    axes[1, i].axis('off')
plt.show()

# Print unique values in images and masks
print("Unique values in train images:", torch.unique(train_images))
print("Unique values in train masks:", torch.unique(train_masks))
print("Unique values in val images:", torch.unique(val_images))
print("Unique values in val masks:", torch.unique(val_masks))

# Move the model to the appropriate device
net.to(device)

# Function to log GPU memory usage
def log_gpu_memory_usage(step):
    current_usage = torch.cuda.memory_allocated()
    max_usage = torch.cuda.max_memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"[Step {step}] Current GPU memory usage: {current_usage / 1024**2:.2f} MB")
    print(f"[Step {step}] Max GPU memory usage: {max_usage / 1024**2:.2f} MB")
    print(f"[Step {step}] Total GPU memory: {total_memory / 1024**2:.2f} MB")
    # Reset the peak memory stats for accurate logging
    torch.cuda.reset_peak_memory_stats()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"[Step {step}] Reserved GPU memory: {reserved_memory / 1024**2:.2f} MB")

epochs = 2

torch.cuda.empty_cache()

scaler = torch.amp.GradScaler("cuda")

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    correct_train_predictions = 0
    total_train_pixels = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for step, (images, masks, _, _) in enumerate(progress_bar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = net(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        log_gpu_memory_usage(step)

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

    # Set the model to evaluation mode

torch.cuda.empty_cache()

net.eval()

# Load a sample image from the dataset
sample_image, sample_mask, _, _ = dataset[50]  # Change the index to load a different sample

# Move the sample image to the appropriate device
sample_image = sample_image.to(device).unsqueeze(0)  # Add batch dimension

# Make a prediction
with torch.no_grad():
    prediction = net(sample_image)

# Convert the prediction to a binary mask
predicted_mask = (prediction > 0.5).float()

criterion = DiceLoss()

# Calculate Dice coefficient
dice_coefficient = 1 - criterion(predicted_mask, sample_mask.to(device))

# Plot the sample image, ground truth mask, and predicted mask
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(sample_image.squeeze().permute(1, 2, 0).cpu().numpy())
ax[0].set_title("Sample Image")
ax[0].axis('off')

ax[1].imshow(sample_mask.squeeze().cpu().numpy(), cmap='gray')
ax[1].set_title("Ground Truth Mask")
ax[1].axis('off')

ax[2].imshow(predicted_mask.squeeze().cpu().numpy(), cmap='gray')
ax[2].set_title(f"Predicted Mask\nDice Coefficient: {dice_coefficient:.4f}")
ax[2].axis('off')

plt.show()

root_dir = "/data/data_gentuity"
testset = OCTDataset(root_dir, train=False, is_gentuity=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

net.eval()
net.to(device)
correct = 0 
test_loss = 0.0 
criterion = DiceLoss()

with torch.no_grad():  # Disable gradient calculation
    for images, masks, _, _ in tqdm(testloader, desc="Testing", leave=False):
        images, masks = images.to(device), masks.to(device)

        outputs = net(images)
        predicted = (outputs > 0.5).float()
        loss = criterion(predicted, masks)
        test_loss += loss.item() * images.size(0)

        # Calculate accuracy
        correct += (predicted == masks).sum().item()

# Calculate average loss and accuracy
test_loss /= len(testloader.dataset)
accuracy = correct / (len(testloader.dataset) * masks.size(1) * masks.size(2) * masks.size(3))

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Load a sample image from the test dataset
sample_image, sample_mask, _, _ = testset[50]  # Change the index to load a different sample

# Move the sample image to the appropriate device
sample_image = sample_image.to(device).unsqueeze(0)  # Add batch dimension

# Make a prediction
with torch.no_grad():
    prediction = net(sample_image)

# Convert the prediction to a binary mask
predicted_mask = (prediction > 0.5).float()

# Plot the sample image, ground truth mask, and predicted mask
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(sample_image.squeeze().permute(1, 2, 0).cpu().numpy())
ax[0].set_title("Sample Image")
ax[0].axis('off')

ax[1].imshow(sample_mask.squeeze().cpu().numpy(), cmap='gray')
ax[1].set_title("Ground Truth Mask")
ax[1].axis('off')

ax[2].imshow(predicted_mask.squeeze().cpu().numpy(), cmap='gray')
ax[2].set_title("Predicted Mask")
ax[2].axis('off')

plt.show()