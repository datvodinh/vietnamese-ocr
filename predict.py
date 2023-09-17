import argparse
from src.utils.inference import Inference
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",required=True)
parser.add_argument("--img_path",required=True)

arg = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
infer = Inference(MODEL_PATH  = arg.model_path,device=device)

print(infer.predict(img_dir=arg.img_path))