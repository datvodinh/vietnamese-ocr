from src.utils.vocab import Vocabulary
from src.utils.generator import OCRDataset
from src.utils.transform import Transform
from src.model.model import OCRTransformerModel
from src.utils.statistic import Statistic
from src.utils.progress_bar import *
from src.utils.lr_scheduler import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

seed_everything(42)
class Trainer:
    def __init__(self,
                 config,
                 MODEL_PATH = None,
                 IMAGE_PATH = None,
                 TARGET_PATH = None):


        self.config     = config
        self.vocabulary = Vocabulary(data_path   = TARGET_PATH,
                                     device      = config['device'])
        self.dataset    = OCRDataset(root_dir    = IMAGE_PATH,
                                     device      = config['device'],
                                     transform   = Transform(img_size = config['encoder']['swin']['img_size']),
                                     target_dict = self.vocabulary.target_dict)
        self.dataloader = DataLoader(dataset     = self.dataset,
                                     batch_size  = config['batch_size'],
                                     shuffle     = True,
                                     num_workers = config['dataloader']['num_workers'])
        self.stat       = Statistic()
        self.criterion  = nn.CrossEntropyLoss()
        self.len_loader = len(self.dataloader)
        self.pro_bar    = TrainProgressBar(self.config['num_epochs'],self.len_loader)
        if MODEL_PATH is not None:
            try:
                data_dict      = torch.load(MODEL_PATH)
                self.model     = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'])
                self.model.load_state_dict(data_dict['state_dict'])
                load_scheduler = True
                self.config    = config
                print('TRAINING CONTINUE!')
            except:
                self.model     = OCRTransformerModel(config,self.vocabulary.vocab_size)
                load_scheduler = False
                print("TRAIN FROM BEGINNING!")
        else:    
            self.model         = OCRTransformerModel(config,self.vocabulary.vocab_size)
            load_scheduler     = False
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
        self.model.train()
        for e in range(1,self.config['num_epochs']+1):
            idx = 0
            for src,target_input, target_output, target_padding in self.dataloader:
                start_time     = time.perf_counter()
                logits         = self.model(src,target_input) # (B,L,V)
                target_padding = target_padding.reshape(-1)
                target_output  = target_output.reshape(-1)
                logits         = logits[target_padding!=0]
                target_output  = target_output[target_padding!=0]
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
                    self.stat.update_acc(acc,torch.sum(target_padding).item())
                    if self.config['print_type'] == 'per_batch':
                        self.pro_bar.step(idx,e,self.stat.loss,self.stat.acc,start_time,printing=True)
                    else:
                        self.pro_bar.step(idx,e,self.stat.loss,self.stat.acc,start_time,printing=False)
            if self.config['print_type'] == 'per_epoch':
                self.pro_bar.step(idx,e,self.stat.loss,self.stat.acc,start_time,printing=True)
            if e % self.config['save_per_epochs']==0 or e==1:
                self._save_checkpoint(e)
            self.stat.reset()
                
    def _save_checkpoint(self):
        save_dict = {
            'state_dict':self.model.state_dict(),
            'config':self.config,
            'vocab_size':self.vocabulary.vocab_size,
            'letter_to_idx': self.vocabulary.letter_to_idx,
            'idx_to_letter': self.vocabulary.idx_to_letter,
            'scheduler': self.scheduler.state_dict()
        }
        try:
            file_path = f"{self.model_path}/model_{self.config['encoder']['type']}_{self.config['num_epochs']}.pt"
            torch.save(save_dict, file_path)
        except:
            file_path = f"{self.model_path}"
            torch.save(save_dict, file_path)

