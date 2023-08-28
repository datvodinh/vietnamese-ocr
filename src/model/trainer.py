from src.utils.vocab import Vocabulary
from src.utils.generator import OCRDataset
from src.utils.transform import Transform
from src.model.model import OCRTransformerModel
from src.utils.statistic import Statistic
from src.utils.progress_bar import CustomProgressBar

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time

class Trainer:
    def __init__(self,
                 config,
                 IMAGE_PATH = None,
                 TARGET_PATH = None):
    
        self.config     = config
        self.vocabulary = Vocabulary(data_path  = TARGET_PATH,
                                    device      = config['device'])
        self.dataset    = OCRDataset(root_dir   = IMAGE_PATH,
                                    device      = config['device'],
                                    transform   = Transform(t_type=config['preprocessing']),
                                    target_dict = self.vocabulary.target_dict)
        
        self.dataloader = DataLoader(self.dataset,config['batch_size'],shuffle=True)
        self.len_loader = len(self.dataloader)
        self.model      = OCRTransformerModel(config,self.vocabulary.vocab_size)
        self.stat       = Statistic()
        self.criterion  = nn.CrossEntropyLoss()
        self.optimizer  = torch.optim.Adam(self.model.parameters(),lr=config['lr'])
        self.pro_bar    = CustomProgressBar(config['num_epochs'],self.len_loader)

    def train(self):
        for e in range(self.config['num_epochs']):
            idx = 0
            for src,target_input, target_output, target_padding, output_padding in self.dataloader:
                start_time     = time.perf_counter()
                logits         = self.model(src,target_input,target_padding) # (B,L,V)
                output_padding = output_padding.reshape(-1)
                target_output  = target_output.reshape(-1)
                loss           = self.criterion(logits[output_padding!=0],target_output[output_padding!=0])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                idx+=1
                self.stat.update_loss(loss.detach().item())
                self.pro_bar.step(idx,e,self.stat.loss,start_time)
            self.stat.reset()
                


