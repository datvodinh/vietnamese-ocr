import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
class Transform:
    def __init__(self,img_size=(64,192),padding=True,training=True) -> None:
        self.padding    = padding
        self.resize_img = A.Resize(img_size[0],img_size[1])
        
        if training:
            if padding:
                self.pad_img = A.PadIfNeeded(min_height=img_size[0],min_width=img_size[1],position=A.PadIfNeeded.PositionType.RANDOM,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255))
            self.transform = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=10,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        A.GaussNoise(10, p=.2),
                        A.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        A.PixelDropout(p=0.5),
                        A.ImageCompression(95, p=.3),
                        A.ToGray(always_apply=True),
                        A.Normalize(),
                        ToTensorV2()
                    ]
                )
            
        else:
            if padding:
                self.pad_img = A.PadIfNeeded(min_height=img_size[0],min_width=img_size[1],position=A.PadIfNeeded.PositionType.CENTER,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255))
            self.transform = A.Compose(
                [
                    A.ToGray(always_apply=True),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
    def __call__(self,img,img_size=(32,64)):
        if self.padding:    
            img = self.pad_img(image=np.asarray(img))['image']
            img = self.resize_img(image=np.asarray(img))['image']
            img = self.transform(image=np.asarray(img))['image']
        else:
            img = self.resize_img(image=np.asarray(img))['image']
            img = self.transform(image=np.asarray(img))['image']
        return img
# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.
        
        
