from torchvision import transforms,datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import torch.nn.functional as F

class OCRDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 transform=None,
                 target_dict=None):
        
        self.root_dir         = root_dir
        self.transform        = transform
        self.image_paths      = os.listdir(root_dir)
        self.target_dict      = target_dict

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        target_input = self.target_dict[self.image_paths[idx]][0:-1]
        target_output = self.target_dict[self.image_paths[idx]][1:]
        
        target_padding = (target_input>2) * 1.0
        output_padding = (target_output>2) * 1.0
        return image,target_input, target_output, target_padding, output_padding
