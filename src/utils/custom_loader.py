import torch
import os
from PIL import Image
import random

class NormalLoader:
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
        self.transform     = transform
        self.target_dict   = vocab.target_dict
        self.batch_size    = batch_size
        self.device        = device
        self.vocab         = vocab
        self.len           = int(len(self.image_dir) / self.batch_size) + 1 if len(self.image_dir)%self.batch_size!=0 else int(len(self.image_dir) / batch_size)
        
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self._generator()
    
    def _generator(self):
        random.shuffle(self.image_dir)
        for i in range(0,len(self.image_dir),self.batch_size):
            start,end = i,i+self.batch_size
            batch_dir = self.image_dir[start:end]
            src = torch.stack([self.transform(img=Image.open(os.path.join(self.root_dir,f)),
                                                img_size=self.img_size) 
                                                for f in batch_dir]).to(self.device)
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

class ClusterImageLoader:
    def __init__(self,
                 root_dir,
                 vocab,
                 batch_size = 32,
                 transform  = None,
                 device     = torch.device('cpu')):
        
        self.root_dir      = root_dir
        self.image_dir     = os.listdir(root_dir)
        self.transform     = transform
        self.target_dict   = vocab.target_dict
        self.batch_size    = batch_size
        self.device        = device
        self.vocab         = vocab
        self.cluster_path  = self._cluster_image_by_ratios()
        self.len           = 0
        for k in self.cluster_path.keys():
            len_key = len(self.cluster_path[k])
            self.len += int(len_key / batch_size) + 1 if len_key%batch_size!=0 else int(len_key / batch_size)
        
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self._generator()
    def _generator(self):
        list_key = list(self.cluster_path.keys())
        random.shuffle(list_key)
        for k in list_key:
            list_path = self.cluster_path[k]
            random.shuffle(list_path)
            for i in range(0,len(list_path),self.batch_size):
                start,end = i,i+self.batch_size
                batch_dir = list_path[start:end]
                src = torch.stack([self.transform(img=Image.open(os.path.join(self.root_dir,f)),
                                                  img_size=(32,32*k)) 
                                                  for f in batch_dir]).to(self.device)
                
                target = [self.target_dict[f] for f in batch_dir]
                target_in,target_out,padding = self._padding(target)
                yield src,target_in,target_out,padding


    def _cluster_image_by_ratios(self):
        cluster_dict = {}
        for f in self.image_dir:
            h,w = Image.open(os.path.join(self.root_dir,f)).size
            r = max(round(w/h),1)
            r = min(r,5)
            if r not in cluster_dict.keys():
                cluster_dict[r] = [f]
            else:
                cluster_dict[r].append(f)
        return cluster_dict

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
    
class ClusterTargetLoader:
    def __init__(self,
                 root_dir,
                 vocab,
                 batch_size = 32,
                 transform  = None,
                 img_size   = (64,128),
                 device     = torch.device('cpu')):
        
        self.root_dir      = root_dir
        self.image_dir     = os.listdir(root_dir)
        self.transform     = transform
        self.target_dict   = vocab.target_dict
        self.batch_size    = batch_size
        self.device        = device
        self.vocab         = vocab
        self.cluster_path  = self._cluster_by_target_len()
        self.len           = 0
        self.img_size      = img_size
        for k in self.cluster_path.keys():
            len_key = len(self.cluster_path[k])
            self.len += int(len_key / batch_size) + 1 if len_key%batch_size!=0 else int(len_key / batch_size)

    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self._generator()
    
    def _generator(self):
        list_key = list(self.cluster_path.keys())
        random.shuffle(list_key)
        for k in list_key:
            list_path = self.cluster_path[k]
            random.shuffle(list_path)
            for i in range(0,len(list_path),self.batch_size):
                start,end = i,i+self.batch_size
                batch_dir = list_path[start:end]
                src = torch.stack([self.transform(img=Image.open(os.path.join(self.root_dir,f)),
                                                  img_size=(self.img_size[0],self.img_size[1])) 
                                                  for f in batch_dir]).to(self.device)
                
                target = [self.target_dict[f] for f in batch_dir]
                target_in,target_out,padding = self._padding(target)
                yield src,target_in,target_out,padding

    def _cluster_by_target_len(self):
        '''
        Cluster images with the same target length, reduce the number of padding needed 
        (padding takes 80% of data when pad max length) 
        '''
        cluster_dict = {}
        for f in self.image_dir:
            target = self.target_dict[f]
            key = min(len(target),15)
            if key not in cluster_dict.keys():
                cluster_dict[key] = [f]
            else:
                cluster_dict[key].append(f)
        return cluster_dict
    
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