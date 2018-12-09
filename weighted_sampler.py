import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from collections import Counter
import time
import settings


def weighted_sample(inputs, weights, sample_num):
    #print(weights[:100])
    sum_weights = sum(weights)

    #sample_width = sum_weights / sample_num
    outputs = []
    samples = []
    
    for i in range(sample_num):
        samples.append(random.random() * sum_weights)
    samples = sorted(samples)

    #print(samples[:10])
    sample_index = 0
    cur_weights = 0

    for i, w in enumerate(weights):
        while sample_index < sample_num and cur_weights + w > samples[sample_index]:
            outputs.append(inputs[i])
            sample_index += 1
        cur_weights += w

    return outputs

def test_performance():
    inputs = [1,2,3,4,5,6,7,8] *1000
    weights = [1, 5, 10, 1, 20, 1, 5, 1]*1000
    bg = time.time()
    results = weighted_sample(inputs, weights, 100000)
    t = time.time() - bg
    print('time:', t)
    counts = Counter()
    counts.update(results)
    print(counts.most_common(8))

def test():
    inputs = [1,2,3,4,5,6,7,8]
    weights = [1, 5, 10, 1, 20, 1, 5, 1]
    bg = time.time()
    results = weighted_sample(inputs, weights, 1000)
    t = time.time() - bg
    print('time:', t)
    counts = Counter()
    counts.update(results)
    print(counts.most_common(8))


train_prob = [
    0.414682, 0.040358, 0.116536, 0.050238, 0.059797,
    0.080877, 0.032441, 0.090821, 0.001706, 0.001448,
    0.000901, 0.035176, 0.022142, 0.017282, 0.034307,
    0.000676, 0.017057, 0.006758, 0.029029, 0.047696,
    0.005536, 0.121556, 0.025811, 0.095424, 0.010363,
    0.264804, 0.010556, 0.000354]

cls_weight = 1 / np.array(train_prob)

def get_weighted_sample(df, sample_num):
    df = shuffle(df)

    df['weights'] = df.Target.map(lambda x: sum([cls_weight[j] for j in [int(i) for i in x.split()]]))
    #print(df.head())

    return weighted_sample(df['Id'].values, df['weights'].values, sample_num)

def test_sampling():
    df = pd.read_csv(settings.TRAIN_LABEL)
    ids = get_weighted_sample(df, 10000)
    print_label_counts(df, ids)

def print_label_counts(df, ids):
    df_selected = df.set_index('Id').loc[ids]

    label_counts = df_selected['Target'].str.split().apply(pd.Series).stack().value_counts()
    print(label_counts)


if __name__ == '__main__':
    test_sampling()