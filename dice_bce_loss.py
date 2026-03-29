import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([3.0]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, inputs, targets, smooth=1e-4):
        targets = targets.float()

        bce_loss = self.bce(inputs, targets)

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2 * intersection + smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + smooth
        )

        dice_loss = 1 - dice.mean()

        total_loss = 0.3 * bce_loss + 0.7 * dice_loss
        return total_loss