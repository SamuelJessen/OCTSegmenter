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

## Define these before running the script
fold_names = ["fold1", "fold2", "fold3", "fold4", "fold5"]
models_list_base = [
    ("MedSAM Frozen", {"model": "MedSam", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/medsam_frozen_bs=6_dicebce.pth"}),
    ("MedSAM UnFrozen", {"model": "MedSam", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/medsam_unfrozen_bs=6_dicebce.pth"}),
    ("AttentionUnet Frozen", {"model": "AttentionUnet", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/attentionUnet_frozen_bs=6_dicebce.pt"}),
    ("AttentionUnet UnFrozen", {"model": "AttentionUnet", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/attentionUnet_unfrozen_bs=6_dicebce.pt"}),
    ("U-Net Frozen", {"model": "Unet", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/unet_frozen_bs=6_dicebce.pt"}),
    ("U-Net UnFrozen", {"model": "Unet", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/unet_unfrozen_bs=6_dicebce.pt"}),
    ("DeepLabV3+ Frozen", {"model": "DeepLabV3+", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/deeplab_frozen_bs=6_dicebce.pt"}),
    ("DeepLabV3+ UnFrozen", {"model": "DeepLabV3+", "checkpoint_path": "/Users/studiesamuel/Library/CloudStorage/OneDrive-Aarhusuniversitet/Deep Learning/checkpoints/deeplab_unfrozen_bs=6_dicebce.pt"}),
]
csv_results_filename = "results_test_trained_on_terumo.csv"
roc_curve_plot_filename = "roc_curve_test.png"
boxplot_filename = "boxplot_test.png"
save_prediction_images_dir = "output_images"  # Directory to save images with predictions
os.makedirs(save_prediction_images_dir, exist_ok=True)

def generate_models_list_for_folds(base_models_list, fold_names):
    models_list_for_folds = {}
    for fold_name in fold_names:
        
        for model_name, model_config in base_models_list:
            if(model_name == "MedSAM"):
                path_string = (f"{model_name} {fold_name}", {**model_config, "checkpoint_path": f"{model_config['checkpoint_path']}_{fold_name}.pth"})
            else:
                path_string = (f"{model_name} {fold_name}", {**model_config, "checkpoint_path": f"{model_config['checkpoint_path']}_{fold_name}.pt"}) 
            
            models_list_for_folds[fold_name] = [path_string]
            
    return models_list_for_folds

# Generate models list for all folds
models_list_for_folds = generate_models_list_for_folds(models_list_base, fold_names)

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



def plot_and_save_roc_curves_from_df(results_df, fold_name):
    """
    Generate and plot ROC curves for multiple models based on DataFrame containing predictions and true labels.
    """
    # Define the number of points on the FPR axis
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []  # True Positive Rates for each model
    aucs = []  # AUCs for each model
    
    # Prepare a plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique model names
    model_names = results_df['Model'].unique()

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        
        # Filter data for the current model
        model_data = results_df[results_df['Model'] == model_name]
        
        # Initialize empty lists to store true labels and predictions
        true_labels = []
        predictions = []

        # Iterate through the data rows and get arrays directly
        for _, row in model_data.iterrows():
            true_label = row['True label']  # This is already an array
            prediction = row['Prediction value']  # This is already an array

            # Ensure that true_label contains only 0 or 1 (binary classification)
            true_label = (true_label > 0).astype(int)  # Convert to binary 0 or 1

            # Append the arrays to the lists
            true_labels.append(true_label)
            predictions.append(prediction)

        # Convert the lists of arrays into single flattened arrays
        true_labels = np.concatenate(true_labels)
        predictions = np.concatenate(predictions)
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Plot the individual ROC curve
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")

        # Interpolate TPR for mean ROC calculation
        tpr_interp = interp1d(fpr, tpr, bounds_error=False, fill_value=0)(mean_fpr)
        tpr_interp[0] = 0.0  # Ensure the curve starts from (0,0)
        tprs.append(tpr_interp)

    # Calculate mean TPR and variability when there are multiple models
    if len(tprs) > 1:
        mean_tpr = np.mean(tprs, axis=0)
    else:
        # If only one model, use that TPR directly
        mean_tpr = tprs[0]

    mean_tpr[-1] = 1.0  # Ensure that the last TPR is 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})", lw=2, alpha=0.8)

    # Plot variability as shaded region
    if len(tprs) > 1:  # Only plot variability if multiple models are present
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label="± 1 std. dev.")

    # Finalize the plot
    ax.plot([0, 1], [0, 1], 'r--', label="Chance")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC Curve with Variability"
    )
    ax.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Save the plot as an image
    plt.savefig(roc_curve_plot_filename + fold_name)


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

    test_dataset = OCTDataset("/data/data_gentuity",
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
    model_names, dice_coeffs, image_ids, probabilities, true_labels = test_models(f"{models_list_base + fold_name}", f"{save_prediction_images_dir + fold_name}")
    
    results_df = pd.DataFrame({
        "Model": model_names,
        "Dice Score": dice_coeffs,
        "Image ID": image_ids,
        "Prediction value": probabilities,
        "True label": true_labels
    })

    # Save the results to a csv file containing the model names and dice scores, and image_ids
    results_df.to_csv(f"{csv_results_filename + fold_name}", index=False)

    # Generate a boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Model", y="Dice Score", data=results_df, hue="Model", whis=[0, 100])
    plt.title("Performance comparison model trained on terumo data tested on gentuity testset")
    plt.ylabel("Dice Similarity Coefficient")
    plt.xticks(rotation=45)
    plt.show()

    # Save the boxplot as an image
    plt.savefig(f"{boxplot_filename + fold_name}")

    # Plot ROC curves for each fold
    plot_and_save_roc_curves_from_df(results_df, fold_name)

    
