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
    img = np.stack(img, axis=-1)
    img = img.transpose((2,0,1))
    return img


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


def get_train_val_loader(batch_size=4, dev_mode=False, val_num=2000):
    df = pd.read_csv(settings.TRAIN_LABEL)
    split_index = int(df.shape[0] * 0.9)
    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]
    df_val = df_val.iloc[:val_num]

    if dev_mode:
        df_train = df_train.iloc[:10]
        df_val = df_val.iloc[:10]

    img_dir = settings.TRAIN_IMG_DIR
    img_ids_train = df_train['Id'].values.tolist()
    labels_train = df_train['Target'].values.tolist()

    dset_train = ImageDataset(img_dir, img_ids_train, labels_train, img_transform=None)
    dloader_train = data.DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dloader_train.num = len(dset_train)

    img_ids_val = df_val['Id'].values.tolist()
    labels_val = df_val['Target'].values.tolist()

    dset_val = ImageDataset(img_dir, img_ids_val, labels_val, img_transform=None)
    dloader_val = data.DataLoader(dset_val, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader_val.num = len(dset_val)

    return dloader_train, dloader_val

def get_test_loader(batch_size=4, dev_mode=False):
    df = pd.read_csv(settings.SAMPLE_SUBMISSION)

    if dev_mode:
        df = df.iloc[:10]
    img_dir = settings.TEST_IMG_DIR
    img_ids = df['Id'].values.tolist()

    dset = ImageDataset(img_dir, img_ids, None, img_transform=None)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader.num = len(dset)
    return dloader

def test_train_loader():
    loader, _ = get_train_val_loader(batch_size=10, dev_mode=True)
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
    loader = get_test_loader(dev_mode=True)
    print(loader.num)
    for img in loader:
        print(img.size())

if __name__ == '__main__':
    #test_train_loader()
    #test_val_loader()
    test_test_loader()
