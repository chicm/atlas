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
import settings


def open_rgby(img_dir, id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(img_dir, id+'_'+color+'.png'), flags).astype(np.float32)/255
           for color in colors]
    return np.stack(img, axis=-1)


class ImageDataset(data.Dataset):
    def __init__(self, img_dir, img_ids, labels=None, img_transform=None):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.labels = labels
        self.img_transform = img_transform
        
    def __getitem__(self, index):
        img = open_rgby(self.img_dir, self.img_ids[index])

        if self.img_transform is not None:
            img = self.img_transform(img)

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


def get_train_loader(batch_size=4, dev_mode=False):
    df = pd.read_csv(settings.TRAIN_LABEL)

    if dev_mode:
        df = df.iloc[:10]
    img_dir = settings.TRAIN_IMG_DIR
    img_ids = df['Id'].values.tolist()
    labels = df['Target'].values.tolist()

    dset = ImageDataset(img_dir, img_ids, labels, img_transform=None)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader.num = len(dset)
    return dloader

def get_test_loader(batch_size=4, dev_mode=False):
    df = pd.read_csv(settings.SAMPLE_SUBMISSION)

    if dev_mode:
        df = df.iloc[:10]
    img_dir = settings.TEST_IMG_DIR
    img_ids = df['Id'].values.tolist()

    dset = ImageDataset(img_dir, img_ids, None, img_transform=None)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader.num = len(dset)
    return dloader

def test_train_loader():
    loader = get_train_loader(batch_size=10, dev_mode=True)
    for i, (img, target) in enumerate(loader):
        print(img.size(), target.size())
        print(img)
        #print(img)
        if i % 1000 == 0:
            print(i)

def test_val_loader():
    loader = get_val_loader()
    for img, target in loader:
        print(img.size(), target)
        print(torch.max(img), torch.min(img))

def test_test_loader():
    loader = get_test_loader(dev_mode=True, tta_index=1)
    print(loader.num)
    for img in loader:
        print(img.size())

if __name__ == '__main__':
    test_train_loader()
    #test_val_loader()
    #test_test_loader()
