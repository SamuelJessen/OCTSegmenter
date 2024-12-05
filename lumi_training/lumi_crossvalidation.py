import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.oct_dataset import OCTDataset
import torch.nn as nn
from utils.lossfunctions import DiceLoss, DiceBCELoss
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, CheckpointConfig
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils.models import UnetNoPretraining
from utils.helper_methods import trial_dirname_creator
from utils.oct_dataset import OCTDataset
from utils.lossfunctions import DiceLoss
from utils.helper_methods import visualize_cv_splits, plot_cv_indices

def train_model_cv(config):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net = UnetNoPretraining().to(device)

    # Select optimizer based on the configuration
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=config["lr"])

    criterion = DiceLoss()

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    root_dir = config["root_dir"]
    folds= config["folds"]
    
    with open(os.path.join(root_dir, "metadata.csv"), "r") as f:
        metadata_df = pd.read_csv(f)
        skf = StratifiedKFold(n_splits=folds)
        splits = list(skf.split(metadata_df, metadata_df["unique_id"]))

    fold = config["fold"]
    train_indices, val_indices = splits[fold]

    train_dataset = OCTDataset(root_dir, indices=train_indices, transform=transform)
    val_dataset = OCTDataset(root_dir, indices=val_indices, transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Train and validate the model
    print(f"Training on fold {fold}")
    
    best_val_loss = float("inf")
    epochs = config["epochs"]
    no_improvement_epochs = 0
    patience = config["patience"]

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        epoch_steps= 0

        for i, data in enumerate(trainloader):
            images, masks, _, _ = data
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            epoch_steps += 1
            if i % 10 == 9:  # print every 10 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                loss.item()))


        # Calculate training loss and accuracy for the epoch
        train_loss = running_loss / len(trainloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")

        # Validation phase
        net.eval()
        val_loss = 0.0

        with torch.no_grad():  # No need to calculate gradients during validation
            for data in valloader:
                images, masks, _, _ = data
                images, masks = images.to(device), masks.to(device)

                outputs = net(images)
                prediction = (outputs > 0.5).float()
                loss = criterion(prediction, masks)
                val_loss += loss.item() * images.size(0)
        
        # Calculate validation loss and accuracy
        val_loss = val_loss / len(valloader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            print(f"Validation loss improved to {val_loss:.4f}. Saving checkpoint.")
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (net.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(
                    {"loss": val_loss, "accuracy": 1 - val_loss},
                    checkpoint=checkpoint,
                )
        else:
            no_improvement_epochs += 1
            print(f"Validation loss did not improve. Best so far: {best_val_loss:.4f}")
        
        if no_improvement_epochs >= patience:
            print(f"Stopping early. No improvement in {patience} epochs.")
            break
        
    print("Finished Training")

def test_best_model(best_result):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    best_trained_model = UnetNoPretraining().to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path, weights_only=True)
    best_trained_model.load_state_dict(model_state)

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    criterion = DiceLoss()

    root_dir = "/data/data_gentuity"

    test_dataset = OCTDataset(root_dir, transform=transform, train=False, is_gentuity=True)
    testloader = DataLoader(test_dataset, batch_size=best_result.config["batch_size"], shuffle=False)

    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in testloader:
            images, masks, _, _ = data
            images, masks = images.to(device), masks.to(device)

            outputs = best_trained_model(images)
            predicted = (outputs > 0.5).float()
            loss = criterion(predicted, masks)
            total_loss += loss.item() * images.size(0)

    # Calculate average loss and accuracy
    total_loss /= len(testloader.dataset)
    accuracy = 1 - loss

    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.4f}")

def main(num_samples, gpus_per_trial, epochs, smoke_test, folds):
    if smoke_test:
        root_dir = "/data/data_terumo_smoke_test"
        with open(os.path.join(root_dir, "metadata.csv"), "r") as f:
            metadata_df = pd.read_csv(f)
            skf = StratifiedKFold(n_splits=folds)
            visualize_cv_splits(metadata_df, n_splits=folds)

    else:
        print("Using full dataset")
    
    config = {
        "root_dir": root_dir,
        "lr": 1e-4,
        "epochs": epochs,
        "smoke_test": smoke_test,
        "batch_size": tune.choice([4]),
        "optimizer": tune.grid_search(["Adam", "SGD", "RMSprop"]),
        "folds": folds,
        "fold": tune.grid_search(list(range(folds))),
        "patience": 10,
        "model": "UnetNoPretraining",
        "loss_function": "DiceLoss",
        "freeze_encoder": False,
    }

    # Define your checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,  # Only keep the best checkpoint
        checkpoint_score_attribute="loss",  # The metric used to determine the best checkpoint
        checkpoint_score_order="min",  # Keep the checkpoint with the lowest loss
    )

    # Define the run config with the checkpoint config
    run_config = RunConfig(checkpoint_config=checkpoint_config, storage_path="/data/ray_results")

    scheduler = ASHAScheduler(
        max_t=config["epochs"],
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model_cv),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            trial_dirname_creator=trial_dirname_creator,
        ),
        param_space=config,
        run_config=run_config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

main(num_samples=1, gpus_per_trial=1, epochs=1, smoke_test=True, folds=5)