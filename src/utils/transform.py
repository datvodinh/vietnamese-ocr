import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import PIL
class Transform:
    def __init__(self,img_size=(64,256),training=True) -> None:
        self.img_size   = img_size
        self.enhance    = Enhance()
        if training:
            self.transform = A.Compose([
                Binarization(img_size=self.img_size), #dict
                Curve(prob=0.5),
                A.PadIfNeeded(min_height=img_size[0],min_width=img_size[1],
                                position=A.PadIfNeeded.PositionType.RANDOM,
                                border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0), rotate_limit=(-15,15),
                    border_mode=0, interpolation=3, value=[0,0,0], p=0.8),
                A.GridDistortion(distort_limit=0.3, border_mode=0, interpolation=3,
                    value=[0,0,0], p=.5),
                A.PixelDropout(dropout_prob=0.01,drop_value=255,p=0.3),
                A.GaussNoise(10, p=.5),
                A.RandomBrightnessContrast(.1, .2, True, p=0.5),
                A.ImageCompression(95, p=.5),
                A.Normalize(mean=(0.,0.,0.),std=(1.,1.,1.)),
                ToTensorV2()
                ])
            
        else:
            self.transform = A.Compose([
                Binarization(img_size=self.img_size),
                A.PadIfNeeded(min_height=img_size[0],min_width=img_size[1],
                                position=A.PadIfNeeded.PositionType.CENTER,
                                border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
                A.Normalize(mean=(0.,0.,0.),std=(1.,1.,1.)),
                ToTensorV2()
                ])
    def __call__(self,img):
        img = np.asarray(self.enhance(img))
        img = self.transform(image=img)['image']
        return (img.float())
# mean_H = 71.9, median_H = 64.
# mean_W = 131.1, median_W = 118.

class Binarization:
    def __init__(self,img_size=(64,256)) -> None:
        self.img_size = img_size
    def __call__(self,image):
        height, width = image.shape
        img = cv2.GaussianBlur(image, (5,5), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
        img = cv2.resize(img,(min(self.img_size[1],int(self.img_size[0]/height*width)),self.img_size[0]))
        img = np.expand_dims(img , axis = 2)
        img = np.concatenate([img, img, img], axis=2)
        return {"image":img}
    
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

class Curve:
    def __init__(self,prob=1.):
        self.prob = prob
    def __call__(self,**img):
        img = img['image']
        if np.random.uniform(0,1) > self.prob:
            return {"image":img}
        
        height, width = img.shape[:2]
        # Generate a meshgrid of the same shape as the image
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ### Design warping 
        # Normalize the coordinates to the range [-1, 1]
        x = (x - (width / 2)) / (width / 2)
        y = (y - (height / 2)) / (height / 2)
        # # Map the distorted coordinates back to the image space
        x = (x + np.sin(y*2)*0.1).astype(np.float32)
        temp=np.random.uniform(0,1)
        curve = np.random.uniform(0.15,0.35)
        if temp > 0.5:
            y = (y + np.cos(x*2)*-curve).astype(np.float32)
        else:
            y = (y + np.cos(x*2)*curve).astype(np.float32)

        x = ((x * (width / 2)) + (width / 2)).astype(np.float32)
        y = ((y * (height / 2)) + (height / 2)).astype(np.float32)

        # Remap the image using the distorted coordinates
        curved_image = cv2.remap(img, x, y, cv2.INTER_LINEAR)

        return {"image":curved_image}