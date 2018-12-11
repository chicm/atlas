import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import cv2
from sklearn.metrics import f1_score
import settings

def print_counts(df):
    counts = df['Target'].str.split().apply(pd.Series).stack().value_counts().to_frame(name='counts')
    counts['classes'] = counts.index.map(lambda x: int(x))
    counts = counts.sort_values(by=['classes'])
    counts['prob'] = counts.counts.map(lambda x: x / df.shape[0])
    print(type(counts), counts)
    print([round(x, 6) for x in counts.prob.values.tolist()])

def check_hpa():
    df1 = pd.read_csv('HPAv18RBGY_wodpl.csv')
    df2 = pd.read_csv('HPAv18RGBY_WithoutUncertain_wodpl.csv')

    print(df1.head())
    print(df2.head())

    id1 = set(df1.Id.values.tolist())
    id2 = set(df2.Id.values.tolist())
    print(len(id1), len(id2), len(id1-id2), len(id1)-len(id2), id1.issuperset(id2))

def check_hpa_img():
    #img = cv2.imread(r'G:\atlas\HPAv18\jpg\4155_967_E9_3_green.jpg', cv2.IMREAD_GRAYSCALE) 
    img = np.array(Image.open(r'G:\atlas\HPAv18\jpg\4155_967_E9_3_green.jpg').convert('L'))
    #img = cv2.imread(r'G:\atlas\train\000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png')#, cv2.IMREAD_GRAYSCALE) 
    print(img[2000:2050, :20])
    print(img.shape)
    print(np.max(img))


def check_train_distribution():
    df = pd.read_csv(settings.TRAIN_LABEL)
    #print(df.head())
    print_counts(df)

    df = shuffle(df, random_state=6)

    split_index = int(df.shape[0] * 0.9)
    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]

    #print_counts(df_val)

def check_hpa_distribution():
    df = pd.read_csv('HPAv18RGBY_WithoutUncertain_wodpl.csv')
    print_counts(df)

def test_f1():
    y =    np.array([1,1,1,1,1,0,0,0,0,0])
    pred = np.array([0,0,0,0,0,0,0,0,0,0])
    score = f1_score(y, pred)
    print(score)
    print('my f1:', my_f1(y, pred))

def my_f1(y, pred):
    tp = np.sum(y*pred)
    precision = tp / (np.sum(pred) + 1e-8)
    recall = tp / (np.sum(y) + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return F1

if __name__ == '__main__':
    #check_hpa()
    #check_hpa_img()
    #check_hpa_distribution()
    test_f1()