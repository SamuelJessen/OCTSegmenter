import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import segmentation_models_pytorch as smp
from ray import train, tune
from ray.air import session
from ray.train import RunConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler

import torch.optim as optim
import torch.nn as nn
from utils.lossfunctions import DiceLoss, DiceBCELoss  
from utils.models import ResNetUNetWithAttention, MedSAM  
from utils.training_validating_testing import train_and_validate_cv, test_best_model  
from utils.helper_methods import trial_dirname_creator
from segment_anything import sam_model_registry
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"


import time
start_time = time.time()


def train_model(config):
    # Extract the trial ID
    trial_id = session.get_trial_id()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config["model"] == "Unet":
        # Initialize model with the hyperparameters from the config
        net = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

        model_state, optimizer_state = torch.load("/data/best_checkpoints/first_iteration/unet_unfrozen_bs=6_dicebce.pt", weights_only=True)
        net.load_state_dict(model_state)

        if config["freeze_encoder"]:
            for param in net.encoder.parameters():
                param.requires_grad = False

    elif config["model"] == "DeepLabV3+":
        net = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        model_state, optimizer_state = torch.load("/data/best_checkpoints/first_iteration/deeplab_unfrozen_bs=6_dicebce.pt", weights_only=True)
        net.load_state_dict(model_state)

        if config["freeze_encoder"]:
            for param in net.encoder.parameters():
                param.requires_grad = False 
    
    elif config["model"] == "MedSam":
        MedSAM_CKPT_PATH = config["origin_dir"]+"/medsam/medsam_vit_b.pth"
        sam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
        net = MedSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
            ).to(device)
        checkpoint = torch.load("/data/best_checkpoints/first_iteration/medsam_unfrozen_bs=6_dicebce.pth", weights_only=True)
        net.load_state_dict(checkpoint[0])
        
        if config["freeze_encoder"]:
            # Freeze the image encoder
            for param in net.image_encoder.parameters():
                param.requires_grad = False

    elif config["model"] == "AttentionUnet":
        net = ResNetUNetWithAttention()
        model_state, optimizer_state = torch.load("/data/best_checkpoints/first_iteration/attentionUnet_unfrozen_bs=6_dicebce.pt", weights_only=True)
        net.load_state_dict(model_state)

        if(config["freeze_encoder"]):
            net = ResNetUNetWithAttention(freeze_entire_backbone=True)

    net.to(device)

    # Select optimizer based on the configuration
    if config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=config["lr"])

    # Select loss function based on the configuration
    if config["loss_function"] == "DiceLoss":
        criterion = DiceLoss()
    elif config["loss_function"] == "DiceBCELoss":
        criterion = DiceBCELoss()
    elif config["loss_function"] == "BCELoss":
        criterion = nn.BCEWithLogitsLoss()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
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

    train_and_validate_cv(root_dir, config, splits, folds, transform, optimizer, criterion, net, device, trial_id)


def main(num_samples, cpus_per_trial, gpus_per_trial, epochs, smoke_test, folds):
    if smoke_test:
        root_dir = "/data/data_terumo_smoke_test"
        origin_dir = "/data/"

    else:
        root_dir = "/data/data_gentuity"
        origin_dir = "/data/"
    
    config = {
        "root_dir": root_dir,
        "origin_dir": origin_dir,
        "lr": tune.choice([1e-4]),
        "epochs": epochs,
        "smoke_test": smoke_test,
        "batch_size": tune.choice([6]),
        "optimizer": tune.choice(["AdamW"]),
        "folds": folds,
        "patience": 15,
        "loss_function": tune.grid_search(["DiceBCELoss"]),
        #"model": tune.grid_search(["AttentionUnet", "Unet", "DeepLabV3+", "MedSam"]),
        "model": tune.grid_search(["DeepLabV3+"]),
        "freeze_encoder": tune.grid_search([False, True]),
        "use_amp": True,
        "cpus_per_trial": cpus_per_trial,
        "lr_patience": 5,
        "fold": tune.grid_search(list(range(folds))),
    }

    # ASHA SCHEDULER, BUT WILL NOT BE USED
    # scheduler = ASHAScheduler(
    #     max_t=5,
    #     grace_period=5,
    #     reduction_factor=2
    # )

    # Define your checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,  # Only keep the best checkpoint
        checkpoint_score_attribute="loss",  # The metric used to determine the best checkpoint
        checkpoint_score_order="min",  # Keep the checkpoint with the lowest loss
    )

    # Define the run config with the checkpoint config
    run_config = RunConfig(checkpoint_config=checkpoint_config, storage_path="/data/ray_results/fifth_training_gentuity")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="dice_loss",
            mode="min",
            num_samples=num_samples,
            trial_dirname_creator=trial_dirname_creator,
        ),
        param_space=config,
        run_config=run_config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("dice_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    #test_best_model(best_result, origin_dir)

main(num_samples=1, cpus_per_trial=8, gpus_per_trial=1, epochs=50, smoke_test=False, folds=5)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time spent: {elapsed_time:.2f} seconds")