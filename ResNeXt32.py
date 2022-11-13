from base_model import ImageClassificationBase
import torchvision.models as models
import torch.nn as nn


class PDDrawingPretrainedResnext32(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.network = models.resnext50_32x4d(pretrained=True)
        # Freeze early layers
        for param in self.network.parameters():
            param.requires_grad = False
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)
