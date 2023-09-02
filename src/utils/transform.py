import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np
class Transform:
    def __init__(self,training=True) -> None:
        if training:
            self.transform = alb.Compose([
                        alb.Resize(64,128),
                        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(0., 0.15), rotate_limit=1,
                            border_mode=0, interpolation=3, value=[255, 255, 255], p=0.7),
                        alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        alb.GaussNoise(10, p=.2),
                        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        alb.ImageCompression(95, p=.3),
                        alb.ToGray(always_apply=True),
                        alb.Normalize(),
                        # alb.Sharpen()
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = alb.Compose(
                [
                    alb.Resize(64, 128),
                    alb.ToGray(always_apply=True),
                    alb.Normalize(),
                    # alb.Sharpen()
                    ToTensorV2(),
                ]
            )
    def __call__(self,img):
        return self.transform(image=np.asarray(img))['image']
# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.
        
        
