import os
import random
from src.utils.transform import Enhance
from PIL import Image
import numpy as np
import albumentations as A
import cv2

class ConcatAug:
    def __init__(self,height=64,img_dir=None,target_dir=None) -> None:
        self.height = height
        self.img_dir = img_dir
        self.list_img_dir = os.listdir(img_dir)
        self.enhance = Enhance()
        self.pad = A.PadIfNeeded(min_height=64,min_width=256,border_mode=cv2.BORDER_CONSTANT,value=(0))
        self.rotate = A.SafeRotate(limit=45,interpolation=3,border_mode = cv2.BORDER_CONSTANT,value=(0,0,0),p=0.66)

        with open(target_dir,"r",encoding="utf-8") as f:
            data = f.read()
        new_data = list(map(lambda i:i.split("\t"),data.split("\n")))
        if new_data[-1][0]=="":
            new_data.pop(-1)
        
        self.target_dict   = {x[0]:x[1]for x in new_data}
    def generate(self):
        d1,d2 = random.choice(self.list_img_dir), random.choice(self.list_img_dir)
        img_1 = np.asarray(self.enhance(Image.open(os.path.join(self.img_dir,d1)).convert("L")))
        img_2 = np.asarray(self.enhance(Image.open(os.path.join(self.img_dir,d2)).convert("L")))
        h1,w1 = img_1.shape
        h2,w2 = img_2.shape
        h = min(h1,h2)
        new_img_1 = cv2.resize(img_1,(int(h/h1*w1),h))
        new_img_2 = cv2.resize(img_2,(int(h/h2*w2),h))
        new_img   =  np.concatenate((new_img_1,new_img_2),axis=1)
        new_img = cv2.bitwise_not(new_img)
        new_img = cv2.resize(new_img,(256,64))
        new_img = self.pad(image=new_img)['image']
        new_img = self.rotate(image=new_img)['image']
        new_img = np.expand_dims(new_img , axis = 2)
        new_img = np.concatenate([new_img, new_img, new_img], axis=2)
        target = self.target_dict[d1] + self.target_dict[d2]

        return new_img, target
