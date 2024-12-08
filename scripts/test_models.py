import os
# Make sure i can reference other folders
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from utils.oct_dataset import OCTDataset
from utils.lossfunctions import DiceLoss
from utils.models import ResNetUNetWithAttention, MedSAM
from segment_anything import sam_model_registry
import torchmetrics
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d

os.environ["DATA_PATH"] = "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning"

DATA_DIR = os.getenv("DATA_PATH", "/data")
one_drive_path = "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning"

## Define these before running the script
fold_names = ["fold0", "fold1", "fold2", "fold3", "fold4"]
models_list_fold0 = [
    ("MedSAM Frozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00005_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=True_loss=DiceBCELoss_fold=0/checkpoint_000035/checkpoint.pth"}),
    ("MedSAM UnFrozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00000_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=False_loss=DiceBCELoss_fold=0/checkpoint_000034/checkpoint.pth"}),
    ("AttentionUnet Frozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00005_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=True_loss=DiceBCELoss_fold=0/checkpoint_000039/checkpoint.pt"}),
    ("AttentionUnet UnFrozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00000_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=False_loss=DiceBCELoss_fold=0/checkpoint_000034/checkpoint.pt"}),
    ("U-Net Frozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00005_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=True_loss=DiceBCELoss_fold=0/checkpoint_000049/checkpoint.pt"}),
    ("U-Net UnFrozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00000_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=False_loss=DiceBCELoss_fold=0/checkpoint_000049/checkpoint.pt"}),
    ("DeepLabV3+ Frozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-54-04/trial_93b3f_00005_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=True_loss=DiceBCELoss_fold=0/checkpoint_000049/checkpoint.pt"}),
    ("DeepLabV3+ UnFrozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-54-04/trial_93b3f_00000_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=False_loss=DiceBCELoss_fold=0/checkpoint_000036/checkpoint.pt"}),
]
models_list_fold1 = [
    ("MedSAM Frozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00006_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=True_loss=DiceBCELoss_fold=1/checkpoint_000020/checkpoint.pth"}),
    ("MedSAM UnFrozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00001_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=False_loss=DiceBCELoss_fold=1/checkpoint_000026/checkpoint.pth"}),
    ("AttentionUnet Frozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00006_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=True_loss=DiceBCELoss_fold=1/checkpoint_000049/checkpoint.pt"}),
    ("AttentionUnet UnFrozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00001_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=False_loss=DiceBCELoss_fold=1/checkpoint_000048/checkpoint.pt"}),
    ("U-Net Frozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00006_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=True_loss=DiceBCELoss_fold=1/checkpoint_000049/checkpoint.pt"}),
    ("U-Net UnFrozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00001_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=False_loss=DiceBCELoss_fold=1/checkpoint_000043/checkpoint.pt"}),
    ("DeepLabV3+ Frozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-07_21-15-02/trial_9583d_00000_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=True_loss=DiceBCELoss_fold=1/checkpoint_000049/checkpoint.pt"}),
    ("DeepLabV3+ UnFrozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-54-04/trial_93b3f_00001_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=False_loss=DiceBCELoss_fold=1/checkpoint_000032/checkpoint.pt"}),
]
models_list_fold2 = [
    ("MedSAM Frozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00007_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=True_loss=DiceBCELoss_fold=2/checkpoint_000025/checkpoint.pth"}),
    ("MedSAM UnFrozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00002_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=False_loss=DiceBCELoss_fold=2/checkpoint_000024/checkpoint.pth"}),
    ("AttentionUnet Frozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00007_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=True_loss=DiceBCELoss_fold=2/checkpoint_000046/checkpoint.pt"}),
    ("AttentionUnet UnFrozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00002_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=False_loss=DiceBCELoss_fold=2/checkpoint_000033/checkpoint.pt"}),
    ("U-Net Frozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00007_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=True_loss=DiceBCELoss_fold=2/checkpoint_000049/checkpoint.pt"}),
    ("U-Net UnFrozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00002_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=False_loss=DiceBCELoss_fold=2/checkpoint_000042/checkpoint.pt"}),
    ("DeepLabV3+ Frozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-07_21-15-02/trial_9583d_00001_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=True_loss=DiceBCELoss_fold=2/checkpoint_000041/checkpoint.pt"}),
    ("DeepLabV3+ UnFrozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-54-04/trial_93b3f_00002_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=False_loss=DiceBCELoss_fold=2/checkpoint_000037/checkpoint.pt"}),
]
models_list_fold3 = [
    ("MedSAM Frozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00008_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=True_loss=DiceBCELoss_fold=3/checkpoint_000026/checkpoint.pth"}),
    ("MedSAM UnFrozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00003_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=False_loss=DiceBCELoss_fold=3/checkpoint_000037/checkpoint.pth"}),
    ("AttentionUnet Frozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00008_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=True_loss=DiceBCELoss_fold=3/checkpoint_000038/checkpoint.pt"}),
    ("AttentionUnet UnFrozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00003_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=False_loss=DiceBCELoss_fold=3/checkpoint_000037/checkpoint.pt"}),
    ("U-Net Frozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00008_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=True_loss=DiceBCELoss_fold=3/checkpoint_000049/checkpoint.pt"}),
    ("U-Net UnFrozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00003_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=False_loss=DiceBCELoss_fold=3/checkpoint_000032/checkpoint.pt"}),
    ("DeepLabV3+ Frozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-07_21-15-02/trial_9583d_00002_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=True_loss=DiceBCELoss_fold=3/checkpoint_000049/checkpoint.pt"}),
    ("DeepLabV3+ UnFrozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-54-04/trial_93b3f_00003_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=False_loss=DiceBCELoss_fold=3/checkpoint_000039/checkpoint.pt"}),
]
models_list_fold4 = [
    ("MedSAM Frozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00009_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=True_loss=DiceBCELoss_fold=4/checkpoint_000049/checkpoint.pth"}),
    ("MedSAM UnFrozen", {"model": "MedSam", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_21-26-58/trial_1662d_00004_lr=1.0e-04_opt=AdamW_bs=6_model=MedSam_freeze=False_loss=DiceBCELoss_fold=4/checkpoint_000037/checkpoint.pth"}),
    ("AttentionUnet Frozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00009_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=True_loss=DiceBCELoss_fold=4/checkpoint_000049/checkpoint.pt"}),
    ("AttentionUnet UnFrozen", {"model": "AttentionUnet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-31-06/trial_5e719_00004_lr=1.0e-04_opt=AdamW_bs=6_model=AttentionUnet_freeze=False_loss=DiceBCELoss_fold=4/checkpoint_000044/checkpoint.pt"}),
    ("U-Net Frozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00009_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=True_loss=DiceBCELoss_fold=4/checkpoint_000049/checkpoint.pt"}),
    ("U-Net UnFrozen", {"model": "Unet", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_17-16-42/trial_1ed43_00004_lr=1.0e-04_opt=AdamW_bs=6_model=Unet_freeze=False_loss=DiceBCELoss_fold=4/checkpoint_000042/checkpoint.pt"}),
    ("DeepLabV3+ Frozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-07_21-15-02/trial_9583d_00003_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=True_loss=DiceBCELoss_fold=4/checkpoint_000049/checkpoint.pt"}),
    ("DeepLabV3+ UnFrozen", {"model": "DeepLabV3+", "checkpoint_path": f"{one_drive_path}/data/ray_results/fifth_training_gentuity/train_model_2024-12-06_15-54-04/trial_93b3f_00004_lr=1.0e-04_opt=AdamW_bs=6_model=DeepLabV3+_freeze=False_loss=DiceBCELoss_fold=4/checkpoint_000049/checkpoint.pt"}),
]
all_models_list = [models_list_fold0, models_list_fold1, models_list_fold2, models_list_fold3, models_list_fold4]

test_results_dir_path = f"{one_drive_path}/data/test_results"
os.makedirs(test_results_dir_path, exist_ok=True)
csv_results_filename = test_results_dir_path + "/results_test_trained_on_gentuity"
boxplot_filename = test_results_dir_path + "/boxplot_test_trained_on_gentuity"
save_prediction_images_dir = test_results_dir_path + "/output_images"  # Directory to save images with predictions
os.makedirs(save_prediction_images_dir, exist_ok=True)

def save_image_with_prediction_and_mask(image, predicted, mask, image_id, save_dir, model_name):
    # Convert tensors to numpy arrays
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    predicted_np = predicted.cpu().numpy().squeeze()
    mask_np = mask.cpu().numpy().squeeze()

    # Save the predicted and ground truth masks as images
    predicted_path = os.path.join(save_dir, f"{image_id}_predicted_{model_name}.png")
    mask_path = os.path.join(save_dir, f"{image_id}_mask_{model_name}.png")
    plt.imsave(predicted_path, predicted_np, cmap="gray")
    plt.imsave(mask_path, mask_np, cmap="gray")

def plot_roc_curves_across_folds(results_df):
    """
    Plot and save ROC curves for a single model evaluated across multiple folds.
    Parameters:
        results_df (DataFrame): Contains columns 'Model', 'Fold', 'True label', and 'Prediction value'.
                                Each row represents predictions and labels for a fold.
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 8))
    model_name = results_df['Model'].iloc[0]
    folds = results_df['Fold'].unique()
    
    for fold in folds:
        print(f"Processing Fold: {fold} for Model: {model_name}")
        
        # Filter data for the current fold
        fold_data = results_df[results_df['Fold'] == fold]
        true_labels = np.concatenate(fold_data['True label'].to_numpy())
        predictions = np.concatenate(fold_data['Prediction value'].to_numpy())
        
        # Ensure binary classification for true labels
        true_labels = (true_labels > 0).astype(int)
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Plot ROC for this fold
        ax.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.4f})")
        
        # Interpolate TPR for mean calculations
        tpr_interp = interp1d(fpr, tpr, bounds_error=False, fill_value=0)(mean_fpr)
        tpr_interp[0] = 0.0  # Ensure the curve starts at (0, 0)
        tprs.append(tpr_interp)
    
    # Compute and plot the mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})", lw=2, alpha=0.8)
    
    # Add variability shading if multiple folds exist
    if len(tprs) > 1:
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label="± 1 std. dev.")
    
    # Finalize plot
    ax.plot([0, 1], [0, 1], 'r--', label="Chance")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve for {model_name} Across Folds"
    )
    ax.legend(loc="lower right")
    plt.grid()
    plt.show()
    
    # Save plot
    plt.savefig(f"{model_name}_roc_curve.png")



def test_models(models_list, save_dir):
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize lists to store results
    model_names = []
    dice_coeffs = []
    image_ids = []
    probabilities = []
    true_labels = []

    # Define dataset and transformations
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    test_dataset = OCTDataset(f"{one_drive_path}/data_gentuity",
        transform=transform,
        train=False,
        is_gentuity=True,
    )
    test_loader = DataLoader(torch.utils.data.Subset(test_dataset, range(5)), batch_size=1, shuffle=False, num_workers=4)

    # Define loss function
    criterion = DiceLoss()

    # Loop through models and test each one
    for model_name, model_config in models_list:
        print(f"Testing model: {model_name}")

        # Initialize the model
        if model_config["model"] == "Unet":
            net = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        elif model_config["model"] == "DeepLabV3+":
            net = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        elif model_config["model"] == "MedSam":
            sam_model = sam_model_registry['vit_b'](checkpoint=model_config["checkpoint_path"])
            net = MedSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
            )
    
        elif model_config["model"] == "AttentionUnet":
            net = ResNetUNetWithAttention()

        # Load model checkpoint
        if(model_config["model"] != "MedSam"):
            checkpoint_path = model_config["checkpoint_path"]
            model_state, optimizer_state = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
            net.load_state_dict(model_state)
        
        net.to(device)
        net.eval()

        # Test the model
        model_dice_scores = []
        model_probs = []
        model_true_labels = []

        with torch.no_grad():  # Disable gradient calculation
            for image_id, data in enumerate(test_loader):
                if model_config["model"] == "MedSam":
                    # For MedSAM, process with bounding boxes
                    images, masks, _, _ = data
                    images, masks = images.to(device), masks.to(device)

                    # Get image dimensions
                    batch_size, _, height, width = images.size()

                    # Create bounding boxes covering the entire image
                    bboxes = torch.tensor([[0, 0, width, height]] * batch_size, dtype=torch.float32).unsqueeze(1).to(device)

                    # Predict outputs with bounding boxes
                    outputs = net(images, bboxes)
                    outputs = torch.sigmoid(outputs)
                    predicted = (outputs > 0.5).float()
                else:
                    # For other models
                    images, masks, _, _ = data
                    images, masks = images.to(device), masks.to(device)

                    outputs = net(images)
                    outputs = torch.sigmoid(outputs)
                    predicted = (outputs > 0.5).float()

                # Calculate loss dice score from torchmetrics
                dice_metric = torchmetrics.Dice()
                dice_score = dice_metric(predicted, masks.int())
                
                model_dice_scores.append(dice_score.item())
                print(f"Dice score: {dice_score.item():.4f}")

                model_probs.append(outputs.cpu().detach().numpy().flatten())
                model_true_labels.append(masks.cpu().detach().numpy().flatten())

                # Save image with predictions and ground truth mask
                save_image_with_prediction_and_mask(images[0], predicted[0], masks[0], image_id, save_dir, model_name)

                # Store image ID for plotting later
                image_ids.append(image_id)

                print(f"Progress: {len(model_dice_scores)} / {len(test_loader)}", end="\r")


        # Store results
        model_names.extend([model_name] * len(model_dice_scores))
        dice_coeffs.extend(model_dice_scores)
        probabilities.extend(model_probs)
        true_labels.extend(model_true_labels)


        print(f"{model_name} - Average Dice accuracy: {sum(model_dice_scores) / len(model_dice_scores):.4f}")
    
    return model_names, dice_coeffs, image_ids, probabilities, true_labels



for fold_name in fold_names:
    if(fold_name == "fold0"):
        model_names, dice_coeffs, image_ids, probabilities, true_labels = test_models(models_list_fold0, f"{save_prediction_images_dir + fold_name}")
    elif(fold_name == "fold1"):
        model_names, dice_coeffs, image_ids, probabilities, true_labels = test_models(models_list_fold1, f"{save_prediction_images_dir + fold_name}")
    elif(fold_name == "fold2"):
        model_names, dice_coeffs, image_ids, probabilities, true_labels = test_models(models_list_fold2, f"{save_prediction_images_dir + fold_name}")
    elif(fold_name == "fold3"):
        model_names, dice_coeffs, image_ids, probabilities, true_labels = test_models(models_list_fold3, f"{save_prediction_images_dir + fold_name}")
    elif(fold_name == "fold4"):
        model_names, dice_coeffs, image_ids, probabilities, true_labels = test_models(models_list_fold4, f"{save_prediction_images_dir + fold_name}")
    
    results_df = pd.DataFrame({
        "Model": model_names,
        "Dice Score": dice_coeffs,
        "Image ID": image_ids,
        "Prediction value": probabilities,
        "True label": true_labels,
        "Fold": fold_name
    })

    # Save the results to a csv file containing the model names and dice scores, and image_ids
    results_df.to_csv(f"{csv_results_filename + fold_name}.csv", index=False)

    # Generate a boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model", y="Dice Score", data=results_df, hue="Model", whis=[0, 100])
    plt.title(f"Performance comparison train iteration 2 {fold_name}")
    plt.ylabel("Dice Similarity Coefficient")
    plt.xticks(rotation=45)
    plt.show()

    # Save the boxplot as an image
    plt.savefig(f"{boxplot_filename + fold_name}.png")


# Create a big dataframe with all the results from all folds
results_dfs = []
for fold_name in fold_names:
    results_df = pd.read_csv(f"{csv_results_filename + fold_name}.csv")
    results_dfs.append(results_df)

# Create 8 dataframes with the results of each model
model_dfs = {}
for model_name in results_df['Model'].unique():
    model_dfs[model_name] = pd.concat([df[df['Model'] == model_name] for df in results_dfs], ignore_index=True)
    plot_roc_curves_across_folds(model_dfs[model_name])

    
