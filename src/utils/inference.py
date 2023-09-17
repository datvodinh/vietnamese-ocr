from src.model.model import OCRTransformerModel
from src.utils.transform import Transform
from src.utils.progress_bar import EvalProgressBar
from src.utils.transform import Enhance, InvertRescale
import torch
import torch.nn.functional as F
from PIL import Image
import os
import time
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")
class Inference:
    def __init__(self,MODEL_PATH,device=torch.device('cpu')):
        data_dict = torch.load(MODEL_PATH)
        self.device = device
        self.model = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'],device)
        self.model.load_state_dict(data_dict['state_dict'])
        self.model.eval()  
        self.letter_to_idx = data_dict['letter_to_idx']
        self.idx_to_letter = data_dict['idx_to_letter']
        self.config = data_dict['config']
        if self.config['augmentation']:
            self.transform = Transform(img_size=self.config['img_size'],
                                   training=False)
        else:
            self.enhance = Enhance()
            self.transform = A.Compose([
                        InvertRescale(img_size=self.config['img_size']),
                        A.PadIfNeeded(min_height=self.config['img_size'][0],
                                      min_width=self.config['img_size'][1],
                                      position=A.PadIfNeeded.PositionType.CENTER,
                                    border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
                        A.Normalize(mean=(0.,0.,0.),std=(1.,1.,1.)),
                        ToTensorV2()
                    ])

    def predict_batch(self,root_dir,list_dir,batch_size=32,save=False,save_dir=None):
        
        dict_target_decode = self._predict_batch(root_dir,list_dir,batch_size)
        if save:
            self._save(dict_target_decode,save_dir)
        
        return dict_target_decode
    
    def predict(self,**kwargs):
        if "img_dir" in kwargs:
            img_dir              = kwargs["img_dir"]
            new_img              = self._ready_image(img_dir=img_dir)
        else:
            img                  = kwargs["img"]
            new_img              = self._ready_image(img=img)
        dict_img             = {}
        dict_target          = {}
        dict_img["predict"]    = new_img.to(self.device)
        dict_target["predict"] = torch.tensor([0]).long().to(self.device)
        dict_target          = self.model(dict_img,dict_target,mode='predict')
        return self._decode_batch(dict_target)


    def _predict_batch(self,root_dir,list_dir,batch_size):
        len_list_dir = len(list_dir)
        if len_list_dir > 1:
            self.p_bar = EvalProgressBar(total_batches=int(len_list_dir / batch_size)+1 if (len_list_dir%batch_size!=0) else int(len_list_dir / batch_size))
        batch_list_dir = []
        dict_target = {}
        if batch_size == -1:
            batch_list_dir = [list_dir]
        else:
            for i in range(0,len_list_dir,batch_size):
                start = i
                end = min(i+batch_size,len_list_dir)
                batch_list_dir.append(list_dir[start:end])
        c = 0
        for batch_dir in batch_list_dir:
            start_time = time.perf_counter()
            dict_batch_img = {}
            dict_batch_target = {}
            for file_name in batch_dir:
                new_img = self._ready_image(img_dir=os.path.join(root_dir,file_name))
                dict_batch_img[file_name] = new_img.to(self.device)
                dict_batch_target[file_name] = torch.tensor([0]).long().to(self.device)
            dict_batch_target = self.model(dict_batch_img,dict_batch_target,mode='predict')
            dict_target = dict_target | dict_batch_target
            c+=1
            if len_list_dir > 1:
                self.p_bar.step(c,start_time)
        dict_target_decode = self._decode_batch(dict_target)
        return dict_target_decode  

    def _decode_batch(self, dict_target):
        dict_decode = {}
        for k in dict_target.keys():
            chars = [self.idx_to_letter[int(i)] for i in dict_target[k]]
            decoded_chars = [c for c in chars if c not in ['<sos>', '<eos>','<pad>']]
            dict_decode[k] = ''.join(decoded_chars)
        return dict_decode
     
    def _save(self,dict_target_decode,save_dir):
        data = []
        for k in dict_target_decode.keys():
            if len(dict_target_decode[k]) == 0:
                data.append(f"{k} a")
            else:
                data.append(f"{k} {dict_target_decode[k]}")
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = f"{save_dir}/prediction.txt"
        else:
            file_name = "prediction.txt"
        with open(file_name, "w") as f:
            for d in data:
                f.write(d + "\n")
        print(f"Prediction save to: {file_name}")

    def _ready_image(self,**kwargs):
        if "img_dir" in kwargs:
            img = Image.open(kwargs["img_dir"]).convert("L")
        else:
            img = kwargs["img"]
        if self.config['augmentation']:
            new_img = self.transform(img)
        else:
            img = np.asarray(self.enhance(img))
            new_img = self.transform(image=img)['image']

        return new_img