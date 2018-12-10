import os
import argparse
import logging as log
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import scipy.optimize as opt
import numpy as np
from loader import get_train_val_loader, get_hpa_loader
from models import ProteinNet, create_model
import settings

VAL_BATCH_MULTI=8

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

focal_loss = FocalLoss()

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def criterion(outputs, targets):
    #return F.binary_cross_entropy_with_logits(outputs, targets)

    return focal_loss(outputs, targets)


def train(args):
    model, model_file = create_model(args.backbone)

    if args.multi_gpu:
        model = nn.DataParallel(model).cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=0.0001, lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0001, lr=args.lr)

    opt_mode = 'max'
    if args.save_loss:
        opt_mode = 'min'

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode=opt_mode, factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)

    _, val_loader = get_train_val_loader(batch_size=args.batch_size, val_batch_size=args.batch_size*VAL_BATCH_MULTI, val_num=args.val_num)

    model.train()

    iteration = 0
    best_val_score = 0.
    best_val_loss = 10000.

    print('epoch | itr |   lr    |   %             |  loss  |  avg   |  loss  | optim f1 |  best f1  |  thresh  |  time | save |')

    if not args.no_first_val:
        best_val_loss, best_val_score, th = validate(args, model, val_loader, args.batch_size*VAL_BATCH_MULTI)

        print('val   |     |         |                 |        |        | {:.4f} | {:.4f}   |  {:.4f}   |   {:s} |       |'.format(
            best_val_loss, best_val_score, best_val_score, ''))

    if args.val:
        return

    if args.lrs == 'plateau':
        if args.save_loss:
            lr_scheduler.step(best_val_loss)
        else:
            lr_scheduler.step(best_val_score)
    else:
        lr_scheduler.step()
    model.train()

    bg = time.time()
    current_lr = get_lrs(optimizer) 
    for epoch in range(args.epochs):
        train_loss = 0
        if args.hpa:
            train_loader, _ = get_hpa_loader(batch_size=args.batch_size)
        else:
            train_loader, _ = get_train_val_loader(batch_size=args.batch_size, balanced=args.balanced)

        for batch_idx, data in enumerate(train_loader):
            iteration += 1
            x, target = data
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} |{:4d} | {:.5f} | {:7d}/{:7d} | {:.4f} | {:.4f} |'
                    .format(epoch, iteration, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num,
                    loss.item(), train_loss/(batch_idx+1)), end='')

            if iteration % args.iter_save == 0:
                val_loss, val_score, th = validate(args, model, val_loader, args.batch_size*VAL_BATCH_MULTI)
                model.train()
                _save_ckp = ''

                if args.always_save or (val_score > best_val_score and not args.save_loss) or (args.save_loss and val_loss < best_val_loss):
                    if args.multi_gpu:
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                
                best_val_score = max(best_val_score, val_score)
                best_val_loss = min(best_val_loss, val_loss)
                
                if args.lrs == 'plateau':
                    if args.save_loss:
                        lr_scheduler.step(best_val_loss)
                    else:
                        lr_scheduler.step(best_val_score)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer) 

                print(' {:.4f} | {:.4f}   |  {:.4f}   |  {:s} | {:.1f}  | {:4s} |'.format(
                    val_loss, val_score, best_val_score, '', (time.time() - bg) / 60, _save_ckp))
                bg = time.time()

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(np.clip(-x, -100, 100)))

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    #score = np.mean(score)
    return np.mean(score), score

def fit_val(x,y):
    params = 0.5*np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p)[0] - 1.0,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params, maxfev=28000)
    return p

def th_to_str(th):
    return str([round(x,2) for x in th])

def validate(args, model, val_loader, batch_size):
    #print('\nvalidating...')
    model.eval()
    val_loss = 0
    targets = None
    outputs = None
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.cuda(), target.cuda()
            output = model(x)
            loss = criterion(output, target)
            val_loss += loss.item()

            if targets is None:
                targets = target.cpu()
            else:
                targets = torch.cat([targets, target.cpu()])
            if outputs is None:
                outputs = output.cpu()
            else:
                outputs = torch.cat([outputs, output.cpu()])    

    n_batchs = val_loader.num//batch_size if val_loader.num % batch_size == 0 else val_loader.num//batch_size+1
            
    preds = torch.sigmoid(outputs).numpy()
    targets = targets.numpy()

    best_th = fit_val(preds, targets)
    best_th[best_th<0.1] = 0.1
    #print(best_th)
    optimized_score, raw_score = F1_soft(preds, targets, th=best_th)
    if args.val:
        print(raw_score)
        print(best_th)
    #print(optimized_score)
    #print(optimized_score)

    return val_loss / n_batchs, optimized_score, best_th
       
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--optim', choices=['Adam', 'SGD'], type=str, default='SGD', help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=12, type=int, help='lr scheduler patience')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--backbone', default='resnet50', type=str, help='backbone')
    parser.add_argument('--epochs', default=1000, type=int, help='epochs')
    parser.add_argument('--val_num', default=3000, type=int, help='epochs')
    parser.add_argument('--iter_save', default=200, type=int, help='epochs')
    parser.add_argument('--val',action='store_true', help='val only')
    parser.add_argument('--hpa',action='store_true', help='val only')
    parser.add_argument('--pos_weight', default=20, type=int, help='end index of classes')
    parser.add_argument('--tuning_th',action='store_true', help='tuning threshold')
    parser.add_argument('--no_first_val',action='store_true', help='tuning threshold')
    parser.add_argument('--init_ckp', default=None, type=str, help='init checkpoint')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--multi_gpu',action='store_true', help='use multi gpus')
    parser.add_argument('--save_loss',action='store_true', help='use multi gpus')
    parser.add_argument('--balanced',action='store_true', help='use balanced sampler')
    args = parser.parse_args()

    print(args)

    train(args)
