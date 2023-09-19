import torch
import os
import argparse
import yaml
from src.model.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--src",help="Training image path",required=True)
parser.add_argument("--target",help="Training label path",required=True)
parser.add_argument("--model",help="Model save path")
parser.add_argument("--config",help="Config save path")
arg = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if arg.config is not None:
    f_name = arg.config
else:
    f_name = "config/swin_config.yaml"

with open(f_name,"r") as f:
    c = f.read()
config = yaml.safe_load(c)

config['augmentation'] = True
config['print_type'] = "per_batch"

trainer = Trainer(config      = config,
                IMAGE_PATH  = arg.src,
                TARGET_PATH = arg.target,
                MODEL_PATH  = arg.model,
                device      = device)
    
if __name__ == "__main__":
    
    print(config)
    trainer.train()