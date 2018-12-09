import os, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import cv2
import json
from PIL import Image
import random
from sklearn.utils import shuffle
import settings

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightnessContrast,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

class Rotate90(RandomRotate90):
    def apply(self, img, factor=3, **params):
        return np.ascontiguousarray(np.rot90(img, 1))

    #def apply_to_bbox(self, bbox, factor=3, **params):
    #    return F.bbox_rot90(bbox, 3, **params)

def strong_aug(p=1):
    return Compose([
        RandomRotate90(),
        Flip(),
        #Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            #CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        #HueSaturationValue(p=0.3),
    ], p=p)

def augment_inclusive(p=.9):
    return Compose([
        RandomRotate90(),
        Flip(),
        #Transpose(),
        OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
        #
        #HorizontalFlip(.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=.75 ),
        Blur(blur_limit=3, p=.33),
        OpticalDistortion(p=.33),
        GridDistortion(p=.33),
        HueSaturationValue(p=.33)
    ], p=p)

def weak_aug(p=1.):
    return Compose([
        RandomRotate90(),
        Flip(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=.75 ),
        RandomBrightnessContrast(p=0.33),
        #Blur(blur_limit=3, p=.33),
        #OpticalDistortion(p=.33),
        #GridDistortion(p=1.33),
        #HueSaturationValue(p=.33)
    ], p=p)

#def augment(aug, image):
#    return aug(image=image)['image']

def augment_4chan(aug, image):
    #print(image.shape)
    #image[:,:,0:3]=aug(image=image[:,:,0:3])['image']
    #image[:,:,3]=aug(image=image[:,:,1:4])['image'][:,:,2]

    augmented = aug(image=image[:,:,0:3], mask=image[:,:,3])
    image[:,:,0:3] = augmented['image']
    image[:,:,3] = augmented['mask']

    #image[0:3,:,:]=aug(image=image[0:3,:,:])['image']
    #print('>>>', image.shape)
    #print(aug(image=image[1:4,:,:])['image'][2,:,:].shape)
    #image[3,:,:]=aug(image=image[1:4,:,:])['image'][2,:,:]
    return image

def get_tta_aug(tta_index=0):
    tta_augs = {
        1: HorizontalFlip(always_apply=True),
        2: VerticalFlip(always_apply=True),
        3: Compose([HorizontalFlip(always_apply=True),VerticalFlip(always_apply=True)]),
        4: Rotate90(),
        5: Compose([Rotate90(), HorizontalFlip(always_apply=True)]),
        6: Compose([VerticalFlip(always_apply=True), Rotate90()]),
        7: Compose([HorizontalFlip(always_apply=True),VerticalFlip(always_apply=True), Rotate90()]),
    }
    return tta_augs[tta_index]

def open_rgby(img_dir, id, suffix='.png'): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    #flags = cv2.IMREAD_GRAYSCALE
    #img = [cv2.imread(os.path.join(img_dir, id+'_'+color+'.png'), flags).astype(np.float32)/255
    #       for color in colors]
    if suffix == '.png':
        img = [np.array(Image.open(os.path.join(img_dir, id+'_'+color+suffix)).convert('L')) for color in colors]
    else:
        img = [np.array(Image.open(os.path.join(img_dir, id+'_'+color+suffix)).convert('L').resize((512,512))) for color in colors]
    img = np.stack(img, axis=-1)
    #img = img.transpose((2,0,1))
    return img


class ImageDataset(data.Dataset):
    def __init__(self, train_mode, img_dir, img_ids, labels=None, img_transform=None, suffix='.png', tta_index=0):
        self.train_mode = train_mode
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.labels = labels
        self.img_transform = img_transform
        self.suffix = suffix
        self.tta_index = tta_index
        
    def __getitem__(self, index):
        img = open_rgby(self.img_dir, self.img_ids[index], self.suffix)
        #Image.fromarray(img[:,:,0:3], mode='RGB').show()
        #Image.fromarray(img[:,:,3], mode='L').show()
        if self.train_mode:
            #aug = augment_inclusive()
            aug = weak_aug()
            img = augment_4chan(aug, img)
        elif self.tta_index != 0:
            #print(self.tta_index)
            aug = get_tta_aug(self.tta_index)
            #print(aug)
            img = augment_4chan(aug, img)
        else:
            pass
        
        #print(img.shape)
        #Image.fromarray(img[:,:,0:3], mode='RGB').show()
        #Image.fromarray(img[:,:,3], mode='L').show()

        img = img.transpose((2,0,1))
        img = (img /255).astype(np.float32)

        #normalize
        mean = [0.0804, 0.0526, 0.0548, 0.0827]
        std = [0.1496, 0.1122, 0.1560, 0.1497]
        img[0, :,:,] = (img[0, :,:,] - mean[0]) / std[0]
        img[1, :,:,] = (img[1, :,:,] - mean[1]) / std[1]
        img[2, :,:,] = (img[2, :,:,] - mean[2]) / std[2]
        img[3, :,:,] = (img[3, :,:,] - mean[3]) / std[3]

        if self.labels is None:
            return img
        else:
            return img, self.get_label_tensor(self.labels[index])

    def get_label_tensor(self, label):
        classes = set([int(x) for x in label.strip().split()])
        labels = torch.FloatTensor([ 1 if i in classes else 0 for i in range(28)])
        return labels

    def __len__(self):
        return len(self.img_ids)


def get_train_val_loader(batch_size=4, val_batch_size=4, dev_mode=False, val_num=3500):
    df = pd.read_csv(settings.TRAIN_LABEL)
    df = shuffle(df, random_state=6)

    split_index = int(df.shape[0] * 0.9)
    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]
    df_val = df_val.iloc[:val_num]
    print(df_val.shape)

    if dev_mode:
        df_train = df_train.iloc[3:4]
        df_val = df_val.iloc[3:4]

    img_dir = settings.TRAIN_IMG_DIR
    img_ids_train = df_train['Id'].values.tolist()
    labels_train = df_train['Target'].values.tolist()

    dset_train = ImageDataset(True, img_dir, img_ids_train, labels_train, img_transform=None)
    dloader_train = data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader_train.num = len(dset_train)

    img_ids_val = df_val['Id'].values.tolist()
    labels_val = df_val['Target'].values.tolist()

    dset_val = ImageDataset(False, img_dir, img_ids_val, labels_val, img_transform=None)
    dloader_val = data.DataLoader(dset_val, batch_size=val_batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader_val.num = len(dset_val)

    return dloader_train, dloader_val

def get_hpa_loader(batch_size=4, dev_mode=False):
    df_train = pd.read_csv('HPAv18RGBY_WithoutUncertain_wodpl.csv')
    df_train = shuffle(df_train)
    if dev_mode:
        df_train = df_train.iloc[3:4]

    img_dir = settings.HPA_IMG_DIR
    img_ids_train = df_train['Id'].values.tolist()
    labels_train = df_train['Target'].values.tolist()

    dset_train = ImageDataset(True, img_dir, img_ids_train, labels_train, img_transform=None, suffix='.jpg')
    dloader_train = data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader_train.num = len(dset_train)
    return dloader_train, None


def get_test_loader(batch_size=4, dev_mode=False, tta_index=0):
    df = pd.read_csv(settings.SAMPLE_SUBMISSION)

    if dev_mode:
        df = df.iloc[:1]
    img_dir = settings.TEST_IMG_DIR
    img_ids = df['Id'].values.tolist()

    dset = ImageDataset(False, img_dir, img_ids, None, img_transform=None, tta_index=tta_index)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader.num = len(dset)
    return dloader

def test_train_loader():
    loader, _ = get_train_val_loader(batch_size=1, dev_mode=True)
    for i, (img, target) in enumerate(loader):
        print(img.size(), target.size(), torch.max(img))
        print(img)
        break

def test_val_loader():
    loader = get_val_loader()
    for img, target in loader:
        print(img.size(), target)
        print(torch.max(img), torch.min(img))

def test_test_loader(tta_index=0):
    loader = get_test_loader(dev_mode=True, tta_index=tta_index)
    print(loader.num)
    for img in loader:
        print(img.size())

if __name__ == '__main__':
    test_train_loader()
    #test_val_loader()
    #test_test_loader(tta_index=3)
