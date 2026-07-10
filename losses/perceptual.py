import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)

        self.features = nn.Sequential(*list(vgg.features.children())[:16])

        for p in self.features.parameters():
            p.requires_grad = False

        self.loss = nn.L1Loss()

    def forward(self, pred, target):

        if pred.shape[1] == 1:
            pred = pred.repeat(1,3,1,1)

        if target.shape[1] == 1:
            target = target.repeat(1,3,1,1)

        fp = self.features(pred)

        ft = self.features(target)

        return self.loss(fp, ft)
