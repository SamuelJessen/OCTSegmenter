import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)

        # Compute Dice Loss
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)

        return dice_loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs_flatten = inputs.view(-1)
        targets_flatten = targets.view(-1)

        # Calculate intersection and union
        intersection = torch.sum(inputs_flatten * targets_flatten)
        union = torch.sum(inputs_flatten) + torch.sum(targets_flatten)

        # Compute BCE Loss
        bce = self.bce_loss(inputs, targets)

        # Compute Dice Loss
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        Dice_BCE = bce + dice_loss

        return Dice_BCE
