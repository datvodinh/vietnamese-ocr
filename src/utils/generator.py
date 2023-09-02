from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torch.nn.functional as F

class OCRDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 device,
                 transform=None,
                 target_dict=None):
        
        self.root_dir         = root_dir
        self.transform        = transform
        self.image_paths      = os.listdir(root_dir)
        self.target_dict      = target_dict
        self.device           = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        Retrieve an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            image (torch.Tensor): Input image tensor.
            target_input (list): List of integers representing shifted-right target.
            target_output (list): List of integers representing original target.
            target_padding (torch.Tensor): Padding mask for target
        '''
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        target_input = self.target_dict[self.image_paths[idx]][0:-1] # shifted right target
        target_output = self.target_dict[self.image_paths[idx]][1:] # original target
        target_padding = torch.where(((target_input==1) | (target_input==2)),0.,1.) # mask out <eos> <pad>
        
        return image.to(self.device),target_input, target_output, target_padding.to(self.device)
