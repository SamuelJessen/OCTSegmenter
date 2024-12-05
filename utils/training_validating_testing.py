import os
import torch
import neptune
import tempfile
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.lossfunctions import DiceLoss
from utils.oct_dataset import OCTDataset
from utils.models import ResNetUNetWithAttention, MedSAM
from utils.data_augmentation import DataAugmentTransform
from ray import train
from ray.train import Checkpoint
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry
import numpy as np


def train_and_validate(root_dir, config, splits, fold, transform, optimizer, criterion, net, device, trial_id):
    # Initialize Neptune run
    run = neptune.init_run(
        project="OCTAA/OCTSegmenter",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MGU2NGNjMi0yNWE0LTRjNzgtOGNlNS1hZDdkMjJhYzYxMWUifQ==",
        name=f"trial_{str(trial_id)}",
        tags="terumo",
    )  # your credentials

    run["sys/group_tags"].add([
        str(config["model"]),
        f"Freezing: {str(config['freeze_encoder'])}",
        str(config["loss_function"]), 
        str(config["optimizer"]), 
        f"Fold: {str(fold)}"
    ])  # Group tags

    # Log configuration parameters
    run["parameters"] = config

    train_indices, val_indices = splits[fold]

    train_dataset = OCTDataset(root_dir, indices=train_indices, transform=transform)
    val_dataset = OCTDataset(root_dir, indices=val_indices, transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["cpus_per_trial"])
    valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["cpus_per_trial"])

    best_val_loss = float("inf")
    epochs = config["epochs"]
    no_improvement_epochs = 0
    patience = config["patience"]
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=config["lr_patience"])

    if config["use_amp"]:
        scaler = torch.amp.GradScaler("cuda")
    
    torch.cuda.empty_cache()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        epoch_steps= 0

        for i, data in enumerate(trainloader):
            if config["model"] == "MedSam":
                images, masks, _, _ = data
                images, masks = images.to(device), masks.to(device)

                # Get image dimensions
                batch_size, _, height, width = images.size()

                # Create bounding boxes that cover the whole image
                bboxes = torch.tensor([[0, 0, width, height]] * batch_size, dtype=torch.float16).unsqueeze(1).to(device)

                optimizer.zero_grad()
                
                if config["use_amp"]:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = net(images, bboxes)
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()

                    # Check for NaN gradients before clipping or optimizer step
                    nan_gradients = False
                    for param in net.parameters():
                        if torch.isnan(param.grad).any():
                            print("NaN gradients detected!")
                            nan_gradients = True
                            break

                    if nan_gradients:
                        # Zero out gradients to prevent NaNs from affecting future batches
                        net.zero_grad()
                        # Skip the batch and move to the next iteration
                        continue

                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                else:    
                    outputs = net(images, bboxes)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()

            else:
                if config["use_amp"]:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        images, masks, _, _ = data
                        images, masks = images.to(device), masks.to(device)

                        optimizer.zero_grad()
                        outputs = net(images)
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    images, masks, _, _ = data
                    images, masks = images.to(device), masks.to(device)

                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()

            running_loss += loss.item() * images.size(0)

            epoch_steps += 1
            if i % 10 == 9:  # print every 10 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                loss.item()))


        # Calculate training loss and accuracy for the epoch
        train_loss = running_loss / len(trainloader.dataset)
        run["train_loss"].append(train_loss)  # Log training loss to neptune
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")

        # Validation phase
        net.eval()
        val_loss = 0.0
        dice_loss = 0.0

        with torch.no_grad():  # No need to calculate gradients during validation
            for data in valloader:
                if config["model"] == "MedSam":
                    images, masks, _, _ = data
                    images, masks = images.to(device), masks.to(device)

                    # Get image dimensions
                    batch_size, _, height, width = images.size()

                    # Create bounding boxes that cover the whole image
                    bboxes = torch.tensor([[0, 0, width, height]] * batch_size, dtype=torch.float16).unsqueeze(1).to(device)

                    outputs = net(images, bboxes)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

                else:
                    images, masks, _, _ = data
                    images, masks = images.to(device), masks.to(device)

                    outputs = net(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

                # Calculate Dice loss
                dice = DiceLoss()
                loss = dice(outputs, masks)
                dice_loss += loss.item() * images.size(0)
        
        # Calculate validation loss and accuracy
        val_loss = val_loss / len(valloader.dataset)
        avg_dice_loss = dice_loss / len(valloader.dataset)
        scheduler.step(val_loss) # Adjust learning rate based on validation loss
        run["val_loss"].append(val_loss)  # Log validation loss
        run["dice_loss"].append(avg_dice_loss)  # Log Dice loss
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

        if config["model"] == "MedSam":
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pth")
                torch.save(net.state_dict(), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": val_loss, "accuracy": 1 - avg_dice_loss, "dice_loss": avg_dice_loss, "fold": fold},
                    checkpoint=checkpoint,
                )
        else:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (net.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": val_loss, "accuracy": 1 - avg_dice_loss, "dice_loss": avg_dice_loss, "fold": fold},
                    checkpoint=checkpoint,
                )

        # Check if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            print(f"Validation loss improved to {val_loss:.4f}. Saving checkpoint.")
            
        else:
            no_improvement_epochs += 1
            print(f"Validation loss did not improve. Best so far: {best_val_loss:.4f}")
        
        if no_improvement_epochs >= patience:
            print(f"Stopping early. No improvement in {patience} epochs.")
            run["early_stopping"] = True
            break

    run.stop()
    torch.cuda.empty_cache()
    print("Finished Training")


def train_and_validate_cv(root_dir, config, splits, folds, transform, optimizer, criterion, net, device, trial_id):
    for fold in range(folds):
        # Train and validate the model
        print(f"Training on fold {fold+1} out of {folds}")
        
        # Initialize Neptune run
        run = neptune.init_run(
            project="OCTAA/OCTSegmenter",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MGU2NGNjMi0yNWE0LTRjNzgtOGNlNS1hZDdkMjJhYzYxMWUifQ==",
            name=f"trial_{str(trial_id)}",
            tags="gentuity",
        )  # your credentials

        run["sys/group_tags"].add([
            str(config["model"]),
            f"Freezing: {str(config['freeze_encoder'])}",
            str(config["loss_function"]), 
            str(config["optimizer"]), 
            f"Fold: {str(fold)}",
            "Third_training",
        ])  # Group tags

        # Log configuration parameters
        run["parameters"] = config

        train_indices, val_indices = splits[fold]

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

        trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["cpus_per_trial"])
        valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["cpus_per_trial"])

        best_val_loss = float("inf")
        epochs = config["epochs"]
        no_improvement_epochs = 0
        patience = config["patience"]
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=config["lr_patience"])

        if config["use_amp"]:
            scaler = torch.amp.GradScaler("cuda")
        
        torch.cuda.empty_cache()
        
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            epoch_steps= 0

            for i, data in enumerate(trainloader):
                if config["model"] == "MedSam":
                    images, masks, _, _ = data
                    images, masks = images.to(device), masks.to(device)

                    # Get image dimensions
                    batch_size, _, height, width = images.size()

                    # Create bounding boxes that cover the whole image
                    bboxes = torch.tensor([[0, 0, width, height]] * batch_size, dtype=torch.float16).unsqueeze(1).to(device)

                    optimizer.zero_grad()

                    if config["use_amp"]:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = net(images, bboxes)
                            loss = criterion(outputs, masks)
                        scaler.scale(loss).backward()

                        # Check for NaN gradients before clipping or optimizer step
                        nan_gradients = False
                        for param in net.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                print("NaN gradients detected!")
                                nan_gradients = True
                                break

                        if nan_gradients:
                            # Zero out gradients to prevent NaNs from affecting future batches
                            net.zero_grad()
                            # Skip the batch and move to the next iteration
                            continue

                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)

                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    else:    
                        outputs = net(images, bboxes)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad()

                else:
                    if config["use_amp"]:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            images, masks, _, _ = data
                            images, masks = images.to(device), masks.to(device)

                            optimizer.zero_grad()
                            outputs = net(images)
                            loss = criterion(outputs, masks)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    else:    
                        images, masks, _, _ = data
                        images, masks = images.to(device), masks.to(device)

                        optimizer.zero_grad()
                        outputs = net(images)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad()

                running_loss += loss.item() * images.size(0)

                epoch_steps += 1
                if i % 10 == 9:  # print every 10 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    loss.item()))


            # Calculate training loss and accuracy for the epoch
            train_loss = running_loss / len(trainloader.dataset)
            run["train_loss"].append(train_loss)  # Log training loss to neptune
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")

            # Validation phase
            net.eval()
            val_loss = 0.0
            dice_loss = 0.0

            with torch.no_grad():  # No need to calculate gradients during validation
                for data in valloader:
                    if config["model"] == "MedSam":
                        images, masks, _, _ = data
                        images, masks = images.to(device), masks.to(device)

                        # Get image dimensions
                        batch_size, _, height, width = images.size()

                        # Create bounding boxes that cover the whole image
                        bboxes = torch.tensor([[0, 0, width, height]] * batch_size, dtype=torch.float32).unsqueeze(1).to(device)

                        outputs = net(images, bboxes)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item() * images.size(0)

                    else:
                        images, masks, _, _ = data
                        images, masks = images.to(device), masks.to(device)

                        outputs = net(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item() * images.size(0)

                    # Calculate Dice loss
                    dice = DiceLoss()
                    loss = dice(outputs, masks)
                    dice_loss += loss.item() * images.size(0)
            
            # Calculate validation loss and accuracy
            val_loss = val_loss / len(valloader.dataset)
            avg_dice_loss = dice_loss / len(valloader.dataset)
            scheduler.step(val_loss) # Adjust learning rate based on validation loss
            run["val_loss"].append(val_loss)  # Log validation loss
            run["dice_loss"].append(avg_dice_loss)  # Log Dice loss
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

            if config["model"] == "MedSam":
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, "checkpoint.pth")
                    torch.save(net.state_dict(), path)
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(
                        {"loss": val_loss, "accuracy": 1 - avg_dice_loss, "dice_loss": avg_dice_loss, "fold": fold},
                        checkpoint=checkpoint,
                    )
            else:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                    torch.save(
                        (net.state_dict(), optimizer.state_dict()), path
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(
                        {"loss": val_loss, "accuracy": 1 - avg_dice_loss, "dice_loss": avg_dice_loss, "fold": fold},
                        checkpoint=checkpoint,
                    )

            # Check if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                print(f"Validation loss improved to {val_loss:.4f}. Saving checkpoint.")
                
            else:
                no_improvement_epochs += 1
                print(f"Validation loss did not improve. Best so far: {best_val_loss:.4f}")
            
            if no_improvement_epochs >= patience:
                print(f"Stopping early. No improvement in {patience} epochs.")
                run["early_stopping"] = True
                break

        run.stop()
        torch.cuda.empty_cache()
        print("Finished Training")


def test_best_model(best_result, root_dir):
    # Initialize Neptune run
    run = neptune.init_run(
        project="OCTAA/OCTSegmenter",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MGU2NGNjMi0yNWE0LTRjNzgtOGNlNS1hZDdkMjJhYzYxMWUifQ==",
        name="best_model_test",
        tags="gentuity"  
    )  # your credentials

    # Log configuration parameters
    run["parameters"] = best_result.config

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    if best_result.config["model"] == "Unet":
        # Initialize model with the hyperparameters from the config
        net = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            activation="sigmoid",           # output activation (sigmoid for binary segmentation)
        )

    elif best_result.config["model"] == "DeepLabV3+":
        net = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
    
    elif best_result.config["model"] == "MedSam":
        MedSAM_CKPT_PATH = root_dir + "/medsam/medsam_vit_b.pth"
        sam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        net = MedSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
            )

    elif best_result.config["model"] == "AttentionUnet":
        net = ResNetUNetWithAttention()

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path, weights_only=True)
    net.load_state_dict(model_state)
    net.to(device)
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    root_dir = root_dir + "/data_gentuity"

    test_dataset = OCTDataset(root_dir, transform=transform, train=False, is_gentuity=True)
    testloader = DataLoader(test_dataset, batch_size=best_result.config["batch_size"], shuffle=False, num_workers=best_result.config["cpus_per_trial"])

    criterion = DiceLoss()

    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in testloader:
            if best_result.config["model"] == "MedSam":
                images, masks, _, _ = data
                images, masks = images.to(device), masks.to(device)

                # Get image dimensions
                batch_size, _, height, width = images.size()

                # Create bounding boxes that cover the whole image
                bboxes = torch.tensor([[0, 0, width, height]] * batch_size, dtype=torch.float32).unsqueeze(1).to(device)

                outputs = net(images, bboxes)
                predicted = (outputs > 0.5).float()
                loss = criterion(predicted, masks)
                total_loss += loss.item() * images.size(0)

            else:
                images, masks, _, _ = data
                images, masks = images.to(device), masks.to(device)

                outputs = net(images)
                predicted = (outputs > 0.5).float()
                loss = criterion(predicted, masks)
                total_loss += loss.item() * images.size(0)

    # Calculate average loss and accuracy
    total_loss /= len(testloader.dataset)
    accuracy = 1 - loss

    run["test_loss"] = total_loss
    run.stop()
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.4f}")