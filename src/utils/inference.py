from src.model.model import OCRTransformerModel
from src.utils.transform import Transform
from src.utils.progress_bar import EvalProgressBar
import torch
import torch.nn.functional as F
import re
from PIL import Image
import os
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Inference:
    def __init__(self,MODEL_PATH):
        data_dict = torch.load(MODEL_PATH)
        self.model = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'])
        self.model.load_state_dict(data_dict['state_dict'])
        self.model.eval()  
        self.letter_to_idx = data_dict['letter_to_idx']
        self.idx_to_letter = data_dict['idx_to_letter']
        self.transform = Transform(training=False)
        
        print(data_dict['config'])

    def predict(self,root_dir,list_dir,batch_size=32,save=False,save_dir=None):
        len_list_dir = len(list_dir)
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
            for d in batch_dir:
                img = Image.open(os.path.join(root_dir,d))
                new_img = self.transform(img)
                file_name = d
                dict_batch_img[file_name] = new_img.to(device)
                dict_batch_target[file_name] = torch.tensor([0]).long().to(device)
            dict_batch_target = self.model(dict_batch_img,dict_batch_target,mode='predict')
            dict_target = dict_target | dict_batch_target
            c+=1
            self.p_bar.step(c,start_time)
        dict_target_decode = self._decode_batch(dict_target)
        
        if save:
            data = []
            for k in dict_target_decode.keys():
                if len(dict_target_decode[k]) == 0:
                    data.append(f"{k} a")
                else:
                    data.append(f"{k} {dict_target_decode[k]}")
            if save_dir is not None:
                file_name = save_dir
            else:
                file_name = "prediction.txt"
            with open(file_name, "w") as f:
                for d in data:
                    f.write(d + "\n")
            print(f"Prediction save to: {file_name}")
        
        return dict_target_decode  

    def _decode_batch(self, dict_target):
        dict_decode = {}
        for k in dict_target.keys():
            chars = [self.idx_to_letter[int(i)] for i in dict_target[k]]
            decoded_chars = [c for c in chars if c not in ['<sos>', '<eos>','<pad>']]
            dict_decode[k] = ''.join(decoded_chars)
        return dict_decode
