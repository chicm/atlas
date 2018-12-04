import os
import glob
import torch
import pandas as pd
import numpy as np
import json

import settings
from utils import get_classes


def get_gjx_classes():
    df = pd.read_csv(r'F:\BaiduYunDownload\model_scores\label_index.csv', names=['classes'])
    gjx_classes = df['classes'].values.tolist()
    gjx_classes = [x.replace(' ', '_') for x in gjx_classes]
    return gjx_classes

def check_classes():
    classes, _ = get_classes()
    gjx_classes = get_gjx_classes()
    for i in range(len(classes)):
        if classes[i] != gjx_classes[i]:
            print(i, classes[i], gjx_classes[i])

def convert_gjx_csvs(csv_dir=r'F:\BaiduYunDownload\model_scores'):
    gjx_classes = get_gjx_classes()
    print(len(gjx_classes))
    convert_dir = os.path.join(csv_dir, 'gjx')
    for file_name in glob.glob(os.path.join(csv_dir, 'scores', '*new.csv')):
        print(file_name)
        df = pd.read_csv(file_name)
        print('eval...')
        df['score'] = df['score'].map(lambda x: eval(','.join(x.replace('\n', '').split())))
        print(df.head())
        for i, c in enumerate(gjx_classes):
            #print(i)
            df[c] = df['score'].map(lambda x: x[i])
            #df[c] = df['score'].map(lambda x: ','.join(x.replace('\n', '').split()))
            
        col_names = ['key_id', *gjx_classes]
        df.to_csv(os.path.join(convert_dir, os.path.basename(file_name)), index=False, columns=col_names)

def ensemble_csvs(csv_files, weights, sub_file):
    print(csv_files)
    classes, _ = get_classes()
    results = []
    for filename in csv_files:
        print(filename)
        df = pd.read_csv(filename)
        df = df[classes]
        results.append(df.values)
    outputs = np.average(results, axis=0, weights=weights)
    outputs = torch.from_numpy(outputs)
    _, preds = outputs.topk(3, 1, True, True)
    preds = preds.numpy()

    create_submission(preds, sub_file)

def create_submission(preds, outfile):
    classes, _ = get_classes()
    label_names = []
    for row in preds:
        label_names.append(' '.join([classes[i] for i in row]))

    meta = pd.read_csv(settings.SAMPLE_SUBMISSION)
    meta['word'] = label_names
    meta.to_csv(outfile, index=False)

if __name__ == '__main__':
    #check_classes()
    #convert_gjx_csvs()

    if True:
        csv_files1 = glob.glob(r'F:\BaiduYunDownload\model_scores\gjx\*.csv')
        csv_files2 = glob.glob(r'F:\BaiduYunDownload\yyqing\*.csv')
        csv_files3 = glob.glob(r'F:\BaiduYunDownload\chicm\*.csv')
        csv_files = csv_files1 + csv_files2 + csv_files3
        w = [1.]*8 + [0.5]*5 +[0.5]*2
        ensemble_csvs(csv_files, weights=w, sub_file='sub/weighted_ensemble_1201_3.csv')