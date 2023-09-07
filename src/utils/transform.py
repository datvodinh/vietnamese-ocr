import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
class Transform:
    def __init__(self,img_size=(64,128),training=True) -> None:
        if training:
            self.transform = A.Compose([
                        A.Resize(img_size[0],img_size[1]),
                        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=15,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        A.GaussNoise(10, p=.2),
                        A.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        A.ImageCompression(95, p=.3),
                        A.ToGray(always_apply=True),
                        A.Normalize(),
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(img_size[0], img_size[1]),
                    A.ToGray(always_apply=True),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
    def __call__(self,img):
        return self.transform(image=np.asarray(img))['image']
# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.
        
        
