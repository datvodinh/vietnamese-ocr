from src.utils.vocab import Vocabulary
from src.utils.transform import Transform
from src.model.model import OCRTransformerModel
from src.utils.statistic import Statistic
from src.utils.progress_bar import *
from src.utils.lr_scheduler import CosineAnnealingWarmupRestarts
from src.utils.dataloader import DataLoader
from src.utils.cer import char_error_rate
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torch.nn as nn
import time
import os
import warnings
import random
import numpy as np
import cv2
from PIL import Image
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self,
                 config,
                 MODEL_PATH  = None,
                 IMAGE_PATH  = None,
                 TARGET_PATH = None,
                 device      = torch.device('cpu')):
        seed_everything(config['seed'])
        self.device     = device
        self.config     = config
        self.vocabulary = Vocabulary(data_path   = TARGET_PATH,
                                     device      = device)
        if "augmentation" in config:
            if config['augmentation']:
                self.transform  = Transform(img_size = config['img_size'],
                                        training = True)
                self.eval_transform = Transform(img_size = config['img_size'],
                                        training = False)
            else:
                self.transform = None
                self.eval_transform = None
        else:
            self.transform = None
            self.eval_transform = None
            
        self.dataloader = DataLoader(root_dir = IMAGE_PATH,
                                    vocab       = self.vocabulary,
                                    batch_size  = config['batch_size'],
                                    img_size    = config['img_size'],
                                    transform   = self.transform,
                                    device      = device)
        
        self.stat       = Statistic()
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        self.len_loader = len(self.dataloader)
        self.pro_bar    = TrainProgressBar(self.config['num_epochs'],self.len_loader)
        if MODEL_PATH is not None:
            try:
                data_dict      = torch.load(MODEL_PATH)
                self.model     = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'],device)
                self.model.load_state_dict(data_dict['state_dict'])
                load_scheduler = True
                self.config    = data_dict['config']
                self.cer_val   = data_dict['cer_val']
                print('TRAINING CONTINUE!')
            except:
                self.model     = OCRTransformerModel(config,self.vocabulary.vocab_size,device)
                load_scheduler = False
                self.cer_val   = 100
                print("TRAIN FROM BEGINNING!")
        else:    
            self.model         = OCRTransformerModel(config,self.vocabulary.vocab_size,device)
            load_scheduler     = False
            self.cer_val       = 100
            print("TRAIN FROM BEGINNING!")
        self.optimizer  = torch.optim.AdamW(self.model.parameters(),lr=self.config['lr'], betas=(0.9, 0.98), eps=1e-09)

        self.scheduler = OneCycleLR(optimizer=self.optimizer,
                                    total_steps=400000,
                                    max_lr=config['scheduler']['max_lr'],
                                    pct_start=0.1)
        if load_scheduler:
            self.scheduler.load_state_dict(data_dict['scheduler'])
        self.model_path = MODEL_PATH
        if ".pt" not in self.model_path:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        
        
    def train(self):
        for e in range(1,self.config['num_epochs']+1):
            self.model.train()
            idx = 0
            for src,target_input, target_output, target_padding in self.dataloader:
                start_time     = time.perf_counter()
                logits         = self.model(src,target_input, target_padding) # (B,L,V)
                target_padding = target_padding.reshape(-1)
                target_output  = target_output.reshape(-1)
                logits         = logits[target_padding==False]
                target_output  = target_output[target_padding==False]
                loss           = self.criterion(logits,target_output)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.config["max_grad_norm"])
                self.optimizer.step()
                if self.config['scheduler']['active']:
                    self.scheduler.step()
                with torch.no_grad():
                    acc = torch.mean((torch.argmax(logits,dim=1)==target_output).float())
                    idx+=1
                    self.stat.update_loss(loss.detach().item())
                    self.stat.update_acc(acc,torch.sum(target_padding==False).item())
                    if self.config['print_type'] == 'per_batch':
                        self.pro_bar.step(idx,e,self.stat.loss,self.stat.acc,start_time,printing=True)
                    else:
                        self.pro_bar.step(idx,e,self.stat.loss,self.stat.acc,start_time,printing=False)

            eval_dict = self._eval(self.dataloader.root_dir,self.dataloader.val_dir,self.device)
            pred_list = list(eval_dict.values())
            true_list = [self.vocabulary.decode(self.dataloader.target_dict[k]) for k in eval_dict.keys()]
            cer_score = char_error_rate(pred_list,true_list).item()
            if cer_score <= self.cer_val:
                self.cer_val = cer_score
                self._save_checkpoint(save_best=True)
            print(f"CER: {cer_score:.5f} |")
            
            if self.config['print_type'] == 'per_epoch':
                self.pro_bar.step(idx,e,self.stat.loss,self.stat.acc,start_time,printing=True)
            if e % self.config['save_per_epochs']==0 or e==1:
                self._save_checkpoint()
            self.stat.reset()
                
    def _save_checkpoint(self,save_best=False):
        save_dict = {
            'state_dict':self.model.state_dict(),
            'config':self.config,
            'vocab_size':self.vocabulary.vocab_size,
            'letter_to_idx': self.vocabulary.letter_to_idx,
            'idx_to_letter': self.vocabulary.idx_to_letter,
            'scheduler': self.scheduler.state_dict(),
            'cer_val': self.cer_val
        }
        if not save_best:
            if ".pt" in self.model_path:
                file_path = self.model_path
            else:
                file_path = f"{self.model_path}/model.pt"
            torch.save(save_dict, file_path)

        else:
            if ".pt" in self.model_path:
                file_path = self.model_path
            else:
                file_path = f"{self.model_path}/model_best.pt"
            torch.save(save_dict, file_path)

    def _eval(self,root_dir,eval_img_dir,device):
        self.model.eval()
        batch_list_dir = []
        dict_target = {}
        if len(eval_img_dir) < 200:
            batch_list_dir = [eval_img_dir]
        else:
            for i in range(0,len(eval_img_dir),200):
                start,end = i,i+200
                batch_list_dir.append(eval_img_dir[start:end])
        for batch_dir in batch_list_dir:
            dict_batch_img = {}
            dict_batch_target = {}
            for d in batch_dir:
                
                if self.eval_transform is not None:
                    img =  Image.open(os.path.join(root_dir,d)).convert("L")
                    new_img = self.eval_transform(img)
                else:
                    img =  cv2.imread(os.path.join(root_dir,d))
                    new_img = torch.from_numpy(img) / 255.0
                    new_img = new_img.permute(2,0,1)
                file_name = d
                dict_batch_img[file_name] = new_img.to(device)
                dict_batch_target[file_name] = torch.tensor([0]).long().to(device)
            dict_batch_target = self.model(dict_batch_img,dict_batch_target,mode='predict')
            dict_target = dict_target | dict_batch_target
        dict_target_decode = self._decode_batch(dict_target)
        return dict_target_decode

    def _decode_batch(self, dict_target):
        dict_decode = {}
        for k in dict_target.keys():
            chars = [self.vocabulary.idx_to_letter[int(i)] for i in dict_target[k]]
            decoded_chars = [c for c in chars if c not in ['<sos>', '<eos>','<pad>']]
            dict_decode[k] = ''.join(decoded_chars)
        return dict_decode