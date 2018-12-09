import os
import argparse
import glob
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pandas as pd
import numpy as np

import settings

'''
31000 / 31072
torch.Size([100, 4, 512, 512])
torch.Size([4, 512, 512])
mean: tensor([0.0804, 0.0526, 0.0548, 0.0827], device='cuda:0')
31072
torch.Size([100, 4, 512, 512])
torch.Size([4, 512, 512])
std: tensor([0.1496, 0.1122, 0.1560, 0.1497], device='cuda:0')
'''
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
    #print(img.shape)
    assert img.shape[0] == 512 and img.shape[1] == 512
    return img

class TrainDataset(data.Dataset):
    def __init__(self, img_dir):
        df = pd.read_csv(settings.TRAIN_LABEL)
        self.img_ids = df.Id.values.tolist()
        self.num = len(self.img_ids)
        self.img_dir = img_dir
        #self.img_transforms = transforms.Compose([
        #    transforms.Resize(img_sz),
        #    transforms.ToTensor()
        #    ])

    def __getitem__(self, index):
        img = open_rgby(self.img_dir, self.img_ids[index])
        #img = Image.open(self.file_names[index], 'r')
        #img = img.convert('RGB')
        #img = self.img_transforms(img)
        img = img.transpose((2,0,1))
        img = (img /255).astype(np.float32)

        return img
    def __len__(self):
        return self.num

def calculate_mean(args):
    dset = TrainDataset(settings.TRAIN_IMG_DIR)
    print(len(dset))
    dloader = data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)
    img_sum = None
    for i, img in enumerate(dloader):
        if args.n_batches > 0 and i >= args.n_batches:
            break
        img = img.cuda()
        if img_sum is None:
            img_sum = img
        else:
            img_sum += img
        print('{} / {}'.format(args.batch_size * (i+1), len(dset)), end='\r')
    print('')
    print(img_sum.size())
    img_sum = torch.sum(img_sum, 0)
    print(img_sum.size())
    img_sum = torch.sum(torch.sum(img_sum, 1), 1)

    if args.n_batches == 0:
        num_batches = len(dset) // args.batch_size
    elif args.n_batches > 0:
        num_batches = args.n_batches
    else:
        raise ValueError('n_batches error')
    
    mean = img_sum / (num_batches*args.batch_size*args.img_sz*args.img_sz)
    print('mean:', mean)
    return mean
    

def calculate_std(args, rgb_mean):
    rgb_mean = torch.unsqueeze(rgb_mean, -1)
    rgb_mean = torch.unsqueeze(rgb_mean, -1)
    rgb_mean = torch.unsqueeze(rgb_mean, 0)

    dset = TrainDataset(settings.TRAIN_IMG_DIR)
    print(len(dset))
    dloader = data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)
    std_sum = None
    for i, img in enumerate(dloader):
        if args.n_batches > 0 and i >= args.n_batches:
            break
        img = img.cuda()
        print('{} / {}'.format(args.batch_size * (i+1), len(dset)),end='\r')
        if std_sum is None:
            std_sum = (img - rgb_mean) * (img - rgb_mean)
        else:
            std_sum += (img - rgb_mean) * (img - rgb_mean)
    print(std_sum.size()) 
    std_sum = torch.sum(std_sum, 0)
    print(std_sum.size()) 
    std_sum = torch.sum(torch.sum(std_sum, 1), 1)

    if args.n_batches == 0:
        num_batches = len(dset) // args.batch_size
    elif args.n_batches > 0:
        num_batches = args.n_batches
    else:
        raise ValueError('n_batches error')

    std_dev = torch.sqrt(std_sum / (num_batches*args.batch_size*args.img_sz*args.img_sz - 1))
    print('std:', std_dev)
    return std_dev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--img_sz', default=512, type=int, help='image size')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--n_batches', default=0, type=int, help='batch number')

    args = parser.parse_args()

    rgb_mean = calculate_mean(args)
    calculate_std(args, rgb_mean)
