import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import settings

colors = ['red','green','blue','yellow']

v18_url = 'http://v18.proteinatlas.org/images/'
num_works = 2

IMG_DIR = os.path.join(settings.DATA_DIR, 'HPAv18', 'jpg')

if not os.path.isdir(IMG_DIR):
    os.makedirs(IMG_DIR)

#print(len(imgList))
'''
for i in tqdm(imgList['Id']): # [:5] means downloard only first 5 samples, if it works, please remove it
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path
        r = requests.get(img_url, allow_redirects=True)
        open(DIR + img_name, 'wb').write(r.content)
'''

def download_url(urls):
    num_downloaded = 0
    #print(urls, len(urls))
    for url in urls:
        img = url.split('_')
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = url + "_" + color + ".jpg"
            img_url = v18_url + img_path
            #print(img_url)
            r = requests.get(img_url, allow_redirects=True)
            open(os.path.join(IMG_DIR, img_name), 'wb').write(r.content)
        num_downloaded += 1
        print('downloaded: {}/{}'.format(num_downloaded, len(urls))) 
    print('DONE')

if __name__ == '__main__':
    imgList = pd.read_csv("HPAv18RBGY_wodpl.csv")

    p = Pool(num_works)
    p.map(download_url, np.array_split(imgList['Id'].values, num_works))
