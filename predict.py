import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import settings
from loader import get_test_loader, get_train_val_loader
import cv2
import scipy.optimize as opt
from models import ProteinNet, create_model
from train import validate
from overlap import update_sub, update_with_test_matches


def model_predict(args, model, model_file, check=False, tta_num=1):
    model.eval()

    preds = []
    for flip_index in range(tta_num):
        print('tta index:', flip_index)
        test_loader = get_test_loader(batch_size=args.batch_size, dev_mode=args.dev_mode, tta_index=flip_index)

        outputs = None
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.cuda()
                output = model(x)
                output = F.sigmoid(output)
                if outputs is None:
                    outputs = output.cpu()
                else:
                    outputs = torch.cat([outputs, output.cpu()], 0)
                print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
                if check and i == 0:
                    break

        preds.append(outputs.numpy())
        #return outputs
    #results = torch.mean(torch.stack(preds), 0)
    results = np.mean(preds, 0)
    results_max = np.max(preds, 0)

    parent_dir = model_file+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred_mean_{}.npy'.format(tta_num))
    np_file_max = os.path.join(parent_dir, 'pred_max_{}.npy'.format(tta_num))
    np.save(np_file, results)
    np.save(np_file_max, results_max)

    return results

def predict(args):
    model, model_file = create_model(args.backbone)

    if not os.path.exists(model_file):
        raise AssertionError('model file not exist: {}'.format(model_file))
    _, val_loader = get_train_val_loader(batch_size=args.batch_size, val_batch_size=args.batch_size, val_num=4000)

    model.eval()
    
    #_, preds = outputs.topk(3, 1, True, True)
    _, score, th = validate(args, model, val_loader, args.batch_size)
    print(th)
    print('score:', score)

    if args.val:
        return

    outputs = model_predict(args, model, model_file, tta_num=args.tta_num)

    preds = (outputs > th).astype(np.uint8)
    print(preds.shape)
    
    create_submission(args, preds, args.sub_file)

lb_prob = [
 0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
 0.075268817,0.043841336,0.075268817,0.001700000,0.001400000,
 0.000900000,0.043841336,0.043841336,0.014198783,0.043841336,
 0.000600000,0.028806584,0.014198783,0.028806584,0.059322034,
 0.005000000,0.126126126,0.028806584,0.075268817,0.001000000,
 0.222493880,0.028806584,0.000350000]

train_prob = [
    0.414682, 0.040358, 0.116536, 0.050238, 0.059797,
    0.080877, 0.032441, 0.090821, 0.001706, 0.001448,
    0.000901, 0.035176, 0.022142, 0.017282, 0.034307,
    0.000676, 0.017057, 0.006758, 0.029029, 0.047696,
    0.005536, 0.121556, 0.025811, 0.095424, 0.010363,
    0.264804, 0.010556, 0.000354]

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(np.clip(-x, -100, 100)))

def Count_soft(preds,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    return preds.mean(axis=0)

def fit_test(x,y):
    params = 0.5*np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((Count_soft(x,p) - y,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

def find_lb_th(args, np_file):
    print(np_file)
    preds = np.load(np_file)
    th_t = fit_test(preds, lb_prob)
    th_t[th_t<0.01] = 0.01
    th_t[th_t>0.9] = 0.9
    print(th_t)

    preds = (preds > th_t).astype(np.uint8)
    print(preds.shape)

    create_submission(args, preds, args.sub_file)


def ensemble_np(np_files):
    print(np_files)
    outputs_all = []
    for np_file in np_files:
        outputs_all.append(np.load(np_file))
    outputs = np.mean(outputs_all, 0)
    print(outputs.shape)
    outputs = torch.from_numpy(outputs)
    _, preds = outputs.topk(3, 1, True, True)
    preds = preds.numpy()

    create_submission(args, preds, args.sub_file)

def create_submission(args, preds, outfile):
    label_names = []
    for row in preds:
        label_names.append(' '.join([str(i) for i in range(28) if row[i] == 1]))

    meta = pd.read_csv(settings.SAMPLE_SUBMISSION)
    if args.dev_mode:
        meta = meta.iloc[:len(label_names)]  # for dev mode
    meta['Predicted'] = label_names

    # use leak
    update_sub(meta)
    update_with_test_matches(meta)

    meta.to_csv(outfile, index=False)

def save_raw_csv(np_file):
    df = pd.read_csv(settings.SAMPLE_SUBMISSION)

    np_dir = os.path.dirname(np_file)
    csv_file_name = os.path.join(np_dir, 'raw.csv')
    outputs = np.load(np_file)
    classes, _ = get_classes()

    for i, c in enumerate(classes):
        df[c] = outputs[:, i]
    col_names = ['key_id', *classes]
    df.to_csv(csv_file_name, index=False, columns=col_names)

def create_sub_from_raw_csv(args, csv_file):
    classes, _ = get_classes()
    df = pd.read_csv(csv_file)
    df = df[classes]

    outputs = torch.from_numpy(df.values)
    _, preds = outputs.topk(3, 1, True, True)
    preds = preds.numpy()
    create_submission(args, preds, args.sub_file)

def show_test_img(key_id):
    fn = os.path.join(settings.TEST_SIMPLIFIED_IMG_DIR, '{}.jpg'.format(key_id))
    img = cv2.imread(fn)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Quick Draw')
    parser.add_argument('--backbone', default='resnet50', type=str, help='backbone')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--tta_num', default=1, type=int, help='batch_size')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--sub_file', default='sub/sub1.csv', help='submission file')
    parser.add_argument('--ensemble_np', default=None, type=str, help='np files')
    parser.add_argument('--save_raw_csv', default=None, type=str, help='np files')
    parser.add_argument('--sub_from_csv', default=None, type=str, help='np files')
    parser.add_argument('--find_th', default=None, type=str, help='np files')    
    
    args = parser.parse_args()
    print(args)

    if args.ensemble_np:
        np_files = args.ensemble_np.split(',')
        ensemble_np(np_files)
    elif args.save_raw_csv:
        save_raw_csv(args.save_raw_csv)
    elif args.sub_from_csv:
        create_sub_from_raw_csv(args, args.sub_from_csv)
    elif args.find_th:
        find_lb_th(args, args.find_th)
    else:
        predict(args)

