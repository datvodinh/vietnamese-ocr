from torchvision import transforms
import torch

class Transform:
    train_transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
    ])



    
        
        
