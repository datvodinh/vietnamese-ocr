from loader.vocab import Vocabulary
from loader.generator import OCRDataset
from loader.transform import Transform
from model.model import OCRModel
from model.writer import Writer

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
        self.model = OCRModel(config)
        self.writer = Writer()

        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(),lr=config['lr'])

    def fit(self):
        for _ in range(self.config['num_epochs']):
            for src,target_input, target_output, target_padding, output_padding in self.dataloader:
                logits = self.model(src,target_input,target_padding)
                loss   = self.criterion()


