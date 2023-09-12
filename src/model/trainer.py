from src.utils.vocab import Vocabulary
from src.utils.transform import Transform
from src.model.model import OCRTransformerModel
from src.utils.statistic import Statistic
from src.utils.progress_bar import *
from src.utils.lr_scheduler import CosineAnnealingWarmupRestarts
from src.utils.custom_loader import NormalLoader
from src.utils.cer import char_error_rate
import torch
import torch.nn as nn
import time
import os
import warnings
import random
import numpy as np
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
                 MODEL_PATH = None,
                 IMAGE_PATH = None,
                 TARGET_PATH = None):
        seed_everything(config['seed'])

        self.config     = config
        self.vocabulary = Vocabulary(data_path   = TARGET_PATH,
                                     device      = config['device'])
        self.transform  = Transform(img_size = config['img_size'],
                                    padding  = config['padding'],
                                    enhance  = config['enhancing'],
                                    training = True)
        self.eval_transform = Transform(img_size = config['img_size'],
                                    padding  = config['padding'],
                                    enhance  = config['enhancing'],
                                    training = False)
        self.dataloader = NormalLoader(root_dir = IMAGE_PATH,
                                    vocab       = self.vocabulary,
                                    batch_size  = config['batch_size'],
                                    img_size    = config['img_size'],
                                    transform   = self.transform,
                                    device      = config['device'])
        
        self.stat       = Statistic()
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
        self.len_loader = len(self.dataloader)
        self.pro_bar    = TrainProgressBar(self.config['num_epochs'],self.len_loader)
        if MODEL_PATH is not None:
            try:
                data_dict      = torch.load(MODEL_PATH)
                self.model     = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'])
                self.model.load_state_dict(data_dict['state_dict'])
                load_scheduler = True
                self.config    = data_dict['config']
                self.cer_val   = data_dict['cer_val']
                print('TRAINING CONTINUE!')
            except:
                self.model     = OCRTransformerModel(config,self.vocabulary.vocab_size)
                load_scheduler = False
                self.cer_val   = 100
                print("TRAIN FROM BEGINNING!")
        else:    
            self.model         = OCRTransformerModel(config,self.vocabulary.vocab_size)
            load_scheduler     = False
            self.cer_val       = 100
            print("TRAIN FROM BEGINNING!")
        self.optimizer  = torch.optim.AdamW(self.model.parameters(),lr=self.config['lr'])
        self.scheduler  = CosineAnnealingWarmupRestarts(optimizer         = self.optimizer,
                                                        first_cycle_steps = config['scheduler']['first_cycle_steps'],
                                                        cycle_mult        = config['scheduler']['cycle_mult'],
                                                        max_lr            = config['scheduler']['max_lr'],
                                                        min_lr            = config['scheduler']['min_lr'],
                                                        warmup_steps      = config['scheduler']['warmup_steps'],
                                                        gamma             = config['scheduler']['gamma'])
        if load_scheduler:
            self.scheduler.load_state_dict(data_dict['scheduler'])
        self.model_path = MODEL_PATH
        
        
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

            eval_dict = self._eval(self.dataloader.root_dir,self.dataloader.val_dir,self.config['device'])
            pred_list = list(eval_dict.values())
            true_list = [self.vocabulary.decode(self.dataloader.target_dict[k]) for k in eval_dict.keys()]
            cer_score = char_error_rate(pred_list,true_list).item()
            if cer_score <= self.cer_val:
                self.cer_val = cer_score
                self._save_checkpoint(save_best=True)
            print(f"CER: {cer_score:.2f} |")
            
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
            try:
                file_path = f"{self.model_path}/model_{self.config['encoder']['type']}_{self.config['num_epochs']}.pt"
                torch.save(save_dict, file_path)
            except:
                file_path = f"{self.model_path}"
                torch.save(save_dict, file_path)
        else:
            try:
                file_path = f"{self.model_path}/model_{self.config['encoder']['type']}_{self.config['num_epochs']}_best.pt"
                torch.save(save_dict, file_path)
            except:
                file_path = f"{self.model_path}_best"
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
                img = Image.open(os.path.join(root_dir,d))
                new_img = self.eval_transform(img)
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