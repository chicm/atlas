import os
import argparse
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import settings
from loader import get_train_loader, get_val_loader
import cv2
import ml_metrics as metrics
from models import DrawNet, create_model
from utils import get_classes

MODEL_DIR = settings.MODEL_DIR

criterion = nn.CrossEntropyLoss()

def accuracy(output, label, topk=(1,3,5)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    score = map3(output, label)
    #print('>>', score)
    return res, score

def map3(output, label):
    _, pred = output.topk(3, 1, True, True)
    score = metrics.mapk(label.unsqueeze(-1).cpu().numpy(), pred.cpu().numpy(), 3)
    return score


def train(args):
    print('start training...')
    model, model_file = create_model(args.backbone, args.img_sz)

    if args.multi_gpu:
        model = nn.DataParallel(model).cuda()
    
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('invalid optim')

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    val_loader = get_val_loader(val_num=args.val_num, batch_size=args.batch_size*2, dev_mode=args.dev_mode, img_sz=args.img_sz, workers=args.workers)
    
    best_map3 = 0.

    print('epoch |   lr    |     %        |  loss  |  avg   |  top1  |  top3  |  top5  |  loss  |  map3  |  best  | time |  save  |')

    if not args.no_first_val:
        #best_cls_acc, top1_acc, total_loss, cls_loss, num_loss = f2_validate(args, model, f2_val_loader)#validate_avg(args, model, args.start_epoch)
        top1_acc, top3_acc, top5_acc, val_loss, best_map3 = validate(args, model, val_loader)
        print('val   |         |            |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      |'.format(
            top1_acc, top3_acc, top5_acc, val_loss, best_map3, best_map3))

    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_map3)
    else:
        lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loader = get_train_loader(train_index=epoch % args.train_num, batch_size=args.batch_size, dev_mode=args.dev_mode, img_sz=args.img_sz, workers=args.workers)

        train_loss = 0
        optimizer.zero_grad()

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target = data
            img, target = img.cuda(), target.cuda()
            #optimizer.zero_grad()
            output = model(img)
            
            loss = criterion(output, target)
            loss.backward()
            
            if ((batch_idx+1) % args.acc_grad == 0) or (train_iter > 0 and train_iter % args.iter_val == 0):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                top1_acc, top3_acc, top5_acc, val_loss, map3 = validate(args, model, val_loader)
                _save_ckp = ''
                if args.always_save or map3 > best_map3:
                    best_map3 = map3
                    if args.multi_gpu:
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print('  {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    top1_acc, top3_acc, top5_acc, val_loss, map3, best_map3, (time.time() - bg) / 60, _save_ckp))


                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(map3)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader):
    model.eval()

    total_num = 0
    top1_corrects, top3_corrects, top5_corrects = 0, 0, 0
    total_loss = 0.
    total_map3 = 0.
    n_batches = 0

    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.cuda(), target.cuda()
            output = model(img)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            [top1, top3, top5], map3 = accuracy(output, target)
            top1_corrects += top1
            top3_corrects += top3
            top5_corrects += top5
            total_map3 += map3
            total_num += len(img)
            n_batches += 1
            
    top5_acc = top5_corrects / total_num
    top3_acc = top3_corrects / total_num
    top1_acc = top1_corrects / total_num

    return top1_acc, top3_acc, top5_acc, total_loss/ n_batches, total_map3 / n_batches


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Ship detection')
    parser.add_argument('--backbone', default='resnet18', type=str, help='backbone')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=100, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam', 'rmsprop'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=12, type=int, help='lr scheduler patience')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--train_num', default=95, type=int, help='alway save')
    parser.add_argument('--img_sz', default=256, type=int, help='alway save')
    parser.add_argument('--val_num', default=20000, type=int, help='alway save')
    parser.add_argument('--multi_gpu',action='store_true', help='use multi gpus')
    parser.add_argument('--acc_grad', default=1, type=int, help='gradient accumulation')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    
    args = parser.parse_args()
    print(args)

    train(args)
