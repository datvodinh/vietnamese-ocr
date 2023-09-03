from torchvision import models
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet18()
    def forward(self,x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet50()
    def forward(self,x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x