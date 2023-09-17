import torch
import os
from PIL import Image
import random
import numpy as np
import cv2
class DataLoader:
    def __init__(self,
                 root_dir,
                 vocab,
                 batch_size = 32,
                 img_size   = (64,128),
                 transform  = None,
                 device     = torch.device('cpu')):
        
        self.root_dir      = root_dir
        self.img_size      = img_size
        self.image_dir     = os.listdir(root_dir)
        random.shuffle(self.image_dir)
        self.train_dir     = self.image_dir[:int(0.99 * len(self.image_dir))]
        self.val_dir       = self.image_dir[int(0.99 * len(self.image_dir)):]
        self.transform     = transform
        self.target_dict   = vocab.target_dict
        self.batch_size    = batch_size
        self.device        = device
        self.vocab         = vocab
        self.len           = int(len(self.train_dir) / self.batch_size) + 1 if len(self.train_dir)%self.batch_size!=0 else int(len(self.train_dir) / batch_size)
        
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self._generator()
    
    def _generator(self):
        random.shuffle(self.train_dir)
        for i in range(0,len(self.train_dir),self.batch_size):
            start,end = i,i+self.batch_size
            batch_dir = self.train_dir[start:end]
            if self.transform is not None:
                src = torch.stack([self.transform(img=Image.open(os.path.join(self.root_dir,f)).convert("L")) 
                                                    for f in batch_dir]).to(self.device)
            else:
                src = torch.stack([torch.from_numpy(cv2.imread(os.path.join(self.root_dir,f))) / 255.0
                                                    for f in batch_dir]).to(self.device)
                if src.shape[1]!=3: # B C H W, C = 3
                    src = src.permute(0,3,1,2)
            target = [self.target_dict[f] for f in batch_dir]
            target_in,target_out,padding = self._padding(target)

            yield src,target_in,target_out,padding
    
    def _padding(self,target):
        max_len = 0
        for t in target:
            if len(t) > max_len:
                max_len = len(t)
        padded_target = []
        for t in target:
            pad = torch.full((max_len-t.shape[0],),2).to(self.device) # 2 is <pad> token
            padded_target.append(torch.cat([t,pad],dim=0)) # (B,X+P)

        padded_target = torch.stack(padded_target)
        target_input = padded_target[:,:-1]
        target_output = padded_target[:,1:]
        return target_input, target_output, (target_output==2).to(self.device)