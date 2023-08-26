from torchvision import transforms
import torch

class Transform:
    train_transform = transforms.Compose([
        transforms.Resize((64,118)),
        transforms.ToTensor(),
    ])

# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.

    
        
        
