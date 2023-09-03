import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from torchvision import models

class VGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.vgg19()
    
    def forward(self,x):
        x = self.model.features(x)
        return x