from src.utils.vocab import Vocabulary
from src.utils.generator import OCRDataset
from src.utils.transform import Transform
from src.model.model import OCRModel
from src.utils.writer import Writer

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Trainer:
    def __init__(self,
                 config,
                 IMAGE_PATH = None,
                 TARGET_PATH = None):
    
        self.config = config
        self.vocabulary = Vocabulary(TARGET_PATH)
        self.dataset = OCRDataset(root_dir=IMAGE_PATH,
                                  transform=Transform.train_transform,
                                  target_dict=self.vocabulary.target_dict)
        
        self.dataloader = DataLoader(self.dataset,config['batch_size'],shuffle=True)
        self.model = OCRModel(config,self.vocabulary.vocab_size)
        self.writer = Writer()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'])

    def train(self):
        for _ in range(self.config['num_epochs']):
            for src,target_input, target_output, target_padding, output_padding in self.dataloader:
                logits         = self.model(src,target_input,target_padding) # (B,L,V)
                output_padding = output_padding.reshape(-1)
                target_output  = target_output.reshape(-1)
                loss           = self.criterion(logits[output_padding!=0],target_output[output_padding!=0])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(loss.detach().item())
                


