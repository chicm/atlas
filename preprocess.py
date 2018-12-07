import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import settings

def print_counts(df):
    counts = df['Target'].str.split().apply(pd.Series).stack().value_counts()
    print(counts)

if __name__ == '__main__':
    df = pd.read_csv(settings.TRAIN_LABEL)
    #print(df.head())
    print_counts(df)

    df = shuffle(df, random_state=6)

    split_index = int(df.shape[0] * 0.9)
    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]

    print_counts(df_val)