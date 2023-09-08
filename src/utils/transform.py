import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
class Transform:
    def __init__(self,training=True) -> None:
        self.to_tensor = ToTensorV2()
        if training:
            self.transform = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=10,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        A.GaussNoise(10, p=.2),
                        A.RandomBrightnessContrast(.05, (-.2, 0.2), True, p=0.2),
                        A.GaussianBlur(blur_limit=5,p=0.3),
                        A.ImageCompression(95, p=.3),
                        A.ToGray(always_apply=True),
                        A.PixelDropout(p=0.5),
                    ]
                )
            
        else:
            self.transform = A.Compose(
                [
                    A.ToGray(always_apply=True),
                    # ToTensorV2(),
                ]
            )
    def __call__(self,img,img_size=(32,64)):
        img = A.Resize(img_size[0],img_size[1])(image=np.asarray(img))['image'] # resize height to 64, fixed scale
        img = self.transform(image=np.asarray(img))['image']
        img = self.to_tensor(image=img)['image'] / 255
        return img
# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.
        
        
