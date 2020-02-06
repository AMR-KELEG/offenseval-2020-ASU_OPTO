import re
import os
import glob
import pandas as pd
from pathlib import Path

import random
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def seed_all(seed_value=42):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def clean_text(text):
    # Replace line feed token with a space
    text = re.sub('<LF>', ' ', text)

    # Clean two or more repeated character suffixes
    text = re.sub(r'(.)\1{1,}(\W|$)', r'\1 ', text)

    # TODO: Handle emojis
    return text

def load_data(filename):
    with open(filename, 'r') as f:
        lines = [l.strip().split('\t') for l in f.readlines()]
    df = pd.DataFrame(lines[1:], columns=lines[0])
    df.rename(columns={'tweet': 'text', 'subtask_a': 'target'}, inplace=True)
    df.target = (df['target']=='OFF').astype(int)
    df.text = df.text.apply(clean_text)
    return df

def load_dev_test(filename, test_ratio=0.5):
    df = load_data(filename)
    test_limit_index = int(test_ratio * df.shape[0])
    test_df = df[:test_limit_index]
    dev_df = df[test_limit_index:]
    return dev_df, test_df

def load_lev_data():
    filename = 'data/L-HSAB'
    with open(filename, 'r') as f:
        lines = [l.strip().split('\t') for l in f.readlines()]
    df = pd.DataFrame(lines[1:], columns=lines[0])
    df.rename(columns={'Tweet': 'text', 'Class': 'target'}, inplace=True)
    df.target = (df['target']!='normal').astype(int)
    df.text = df.text.apply(clean_text)
    return df

def load_tun_data():
    filename = 'data/T-HSAB.xlsx'
    df = pd.read_excel(filename, header=None)
    df.rename(columns={0: 'text', 1: 'target'}, inplace=True)
    df.target = (df['target']!='normal').astype(int)
    df.text = df.text.apply(clean_text)
    return df

if __name__ == '__main__':
	df = load_data('data/offenseval-ar-training-v1.tsv')
	print(df.sample())
