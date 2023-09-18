import argparse
from src.utils.inference import Inference
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",required=False)
parser.add_argument("--type",required=False)
parser.add_argument("--img_path",required=True)

arg = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if arg.model_path is not None:
    infer = Inference(MODEL_PATH  = arg.model_path,device=device)
else:
    infer = Inference(MODEL_PATH  = "./checkpoint/model.pt",device=device)
if arg.type is not None:
    if arg.type == "single":
        print(infer.predict(img_dir=arg.img_path))
    elif arg.type == "batch":
        infer.predict_batch(arg.img_path)
else:
    infer.predict_batch(arg.img_path)
