from typing import Any
from torchvision import transforms
import torch
import torch.nn.functional as F
import cv2
import numpy as np
class SobelTranform(object):
    def __init__(self):
        super().__init__()
        # Define Sobel kernels for x and y directions
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    def __call__(self, img):
        # Apply Sobel operators using convolution
        if img.shape[0]==1:
            gradient_x = F.conv2d(img.unsqueeze(0), self.sobel_x.unsqueeze(0).unsqueeze(0),padding=1)
            gradient_y = F.conv2d(img.unsqueeze(0), self.sobel_y.unsqueeze(0).unsqueeze(0),padding=1)
        else:
            gradient_x = F.conv2d(img.unsqueeze(0), self.sobel_x.unsqueeze(0).unsqueeze(0))
            gradient_y = F.conv2d(img.unsqueeze(0), self.sobel_y.unsqueeze(0).unsqueeze(0)) 
        # Compute magnitude and direction of gradients
        magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        return magnitude.squeeze(0)

class Transform:

    def __init__(self,t_type = 'sobel') -> None:
        if t_type == 'sobel': 
            self.train_transform = transforms.Compose([
                transforms.Resize((64,256)),
                transforms.ToTensor(),
                SobelTranform()
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((64,256)),
                transforms.ToTensor(),
            ])

    def __call__(self,img):
        # return self.train_transform(img)
        img = transforms.Resize((64,256))(img)
        _, thresholded_image = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (transforms.ToTensor()(thresholded_image)==0.).float()
# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.
        
        
