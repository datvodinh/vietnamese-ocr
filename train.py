import argparse
import yaml
from src.model.trainer import Trainer
parser = argparse.ArgumentParser()

parser.add_argument('--model',help='Choose model',default='swin_transformer_v2')
parser.add_argument('--src_path',help='Source path',default='ok')
parser.add_argument('--target_path',help='Target path',default='ok')
# parser.add_argument('--src_path',help='Model path',default='ok')

args = parser.parse_args()
with open("./config/swin_config.yaml","r") as f:
    c = f.read()
config = yaml.load(c,Loader=yaml.FullLoader)
# print(config,type(config))
print(args.model)
print(args.src_path)