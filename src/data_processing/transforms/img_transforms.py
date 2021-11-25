import hashlib
import random
import cv2
import numpy as np
from torchvision import transforms

class ImgTransform:

    ''' Augmentation selection for spatio temporal crops
        Default config is as follows:
            transform_prob: 0.5
            noise_multiplier: 10
            flip: 0.5
            jitter: 0.8
            gray: 0.2
)            noise: 0.3 '''


    def __init__(self, img, config):
        self.config = config
        self.img = img
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
       
        random.seed(random.random())
        min(self.height, self.width)
        self.crop_size = random.randrange(30, min(self.height, self.width) - 10)
        if self.crop_size < 30:
            self.crop_size = 30
        if self.width <= self.crop_size or self.height <= self.crop_size:
            self.x = 0
            self.y = 0
        else:
            self.x = random.randrange(1, self.width - self.crop_size)
            self.y = random.randrange(1, self.height - self.crop_size)
        self.noise_multiply = random.randrange(0, config["noise_multiplier"].get())
        self.make_gray = True if random.random() < config["gray"].get() else False
        self.flip_val = random.randrange(-1, 1)
        self.jitter_val = random.randrange(1, config["jitter"].get())

    def crop(self, img):
        img = img[self.y:self.y + self.crop_size,
                  self.x:self.x + self.crop_size]
        return img

    def noise(self, img, amount):
        gaussian_noise = np.zeros_like(img)
        gaussian_noise = cv2.randn(gaussian_noise, 0, amount)
        img = cv2.add(img, gaussian_noise, dtype=cv2.CV_8UC3)
        return img

    def gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def flip(self, img):

        img = cv2.flip(img, self.flip_val)
        return img

    def blur(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    def gen_hash(self, tbh):
        hash_object = hashlib.md5(tbh.encode())
        return hash_object

    def colour_jitter(self, img):
        H, W, C = img.shape
        noise = np.random.randint(0, self.jitter_val, (H, W))
        zitter = np.zeros_like(img)
        zitter[:, :, 1] = noise
        img = cv2.add(img, zitter)
        return img


    def debug(self):
        print("new crop")
        print(f"gray {self.make_gray}")
        print(f"jitter {self.jitter_val}")
        print(f"flip {self.flip_val}")

    def transform_with_prob(self, img):
        img = self.crop(img)
        img = self.flip(img)
        img = self.blur(img)
        img = self.noise(img, self.noise_multiply)
        img = self.colour_jitter(img)
        if self.make_gray:
            img = self.gray(img) 
        #self.debug()
        return img


class Normaliser:

    '''Converts images to PIL, resizes and normalises for appropriate model'''

    def __init__(self, config):
        self.config = config

    def rgb(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def norm(self, img, mean, std):
        tensorfy = transforms.ToTensor()
        img = tensorfy(img)
        normarfy = transforms.Normalize(mean, std)
        img = normarfy(img).unsqueeze(0)
        return img

    # Imagenet ResNet 50/18
    def img_model(self, img):
        img = self.rgb(img)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = self.norm(img, self.config["image_norm"]["mean"].get(
        ), self.config["image_norm"]["std"].get())
        return img

    def location_model(self, img):
        img = self.rgb(img)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = self.norm(img, self.config["location_norm"]["mean"].get(
        ), self.config["location_norm"]["std"].get())
        return img

    def video_model(self, img):
        img = self.rgb(img)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        img = self.norm(img, self.config["video_norm"]["mean"].get(
        ), self.config["video_norm"]["std"].get())
        return img

    def depth_model(self, img):
        img = self.rgb(img)
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
        img = self.norm(img, self.config["depth_norm"]["mean"].get(
        ), self.config["depth_norm"]["std"].get())
        return img
