import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import PIL

'''(array([0.1785674 , 0.17793148, 0.17970584]), mean
 array([0.27464595, 0.27378836, 0.27584539])) std
 '''
class Transform:
    def __init__(self,img_size=(64,192),padding=True,training=True,enhance=True) -> None:
        if enhance:
            self.enhance    = Enhance()
        self.img_size   = img_size
        self.padding    = padding
        self.is_enc     = enhance
        self.resize_img = A.Resize(img_size[0],img_size[1])
        if training:
            self.rescale = False
            if padding:
                self.pad_img = A.PadIfNeeded(min_height=img_size[0],
                                             min_width=img_size[1],
                                             position=A.PadIfNeeded.PositionType.RANDOM,
                                             border_mode=cv2.BORDER_CONSTANT,
                                             value=(255,255,255))
            self.transform = A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.3, 0.), rotate_limit=15,
                            border_mode=0, interpolation=3, value=[255, 255, 255],rotate_method="ellipse", p=0.5),
                        A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                            value=[255, 255, 255], p=.5),
                        A.GaussNoise(10, p=.5),
                        A.RandomBrightnessContrast(.1, .2, True, p=0.5),
                        A.PixelDropout(p=0.5),
                        A.ImageCompression(95, p=.5),
                        A.ToGray(always_apply=True),
                        A.Normalize(mean=[0.5,0.5,0.5],std=[1,1,1]),
                        ToTensorV2()
                    ]
                )
            
        else:
            self.rescale = True
            if padding:
                self.pad_img = A.PadIfNeeded(min_height=img_size[0],
                                             min_width=img_size[1],
                                             position=A.PadIfNeeded.PositionType.CENTER,
                                             border_mode=cv2.BORDER_CONSTANT,
                                             value=(255,255,255))
            self.transform = A.Compose(
                [
                    A.ToGray(always_apply=True),
                    A.Normalize(mean=[0.5,0.5,0.5],std=[1,1,1]),
                    ToTensorV2(),
                ]
            )
    def __call__(self,img):
        if self.rescale:
            img = resize_keep_ratio(img,self.img_size[0])
        if self.is_enc:
            img = self.enhance(img)
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
class Enhance:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        c = [.1, .7, 1.3]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c+.6)
        img = PIL.ImageEnhance.Sharpness(img).enhance(magnitude)
        img = PIL.ImageOps.autocontrast(img)
        return img

def resize_keep_ratio(img, new_height):
    # Calculate the new width while maintaining the aspect ratio
    width, height = img.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    # Resize the img
    resized_img = img.resize((new_width, new_height))
    return resized_img