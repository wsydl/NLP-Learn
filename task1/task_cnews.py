# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:07:05 2019

@author: pc
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
from collections import Counter

TRAIN_PATH = 'E:/task1/cnews.train.txt'
VAL_PATH = 'E:/task1/cnews.val.txt'
TEST_PATH = 'E:/task1/cnews.test.txt'
VOCAB_SIZE = 5000
MAX_LEN = 600
BATCH_SIZE = 64

def read_file(file_name):
    '''
        读文件
    '''
    file_path = {'train': TRAIN_PATH, 'val': VAL_PATH, 'test': TEST_PATH}
    contents = []
    labels = []
    with open(file_path[file_name], 'r', encoding='utf-8') as f:
        for line in f:
            try:
                labels.append(line.strip().split('\t')[0])
                contents.append(line.strip().split('\t')[1])
            except:
                pass
    data = pd.DataFrame()
    data['text'] = contents
    data['label'] = labels
    return data


def build_vocab(data):
    '''
        构建词汇表，
        使用字符级的表示
    '''
    all_content = []
    for _, text in data.iterrows():
        all_content.extend(text['text'])
    counter = Counter(all_content)
    count_pairs = counter.most_common(VOCAB_SIZE - 1)
    words = [i[0] for i in count_pairs]
    words = ['<PAD>'] + list(words)
    
    return words
        

def read_vocab(words):
    words_id = dict(zip(words, range(len(words))))
    return words_id


def read_category(data):
    '''
       将分类目录固定，转换为{类别: id}表示 
    '''
    category = list(data['label'].drop_duplicates())
    return dict(zip(category, range(len(category))))
    
def to_words(content, words):
    return ' '.join(words[i] for i in content)

def preocess_file(data, words_id, category_id):
    """
        将文件转换为id表示
    """
    content = data['text']
    labels = data['label']
    content_id = []
    label_id = []
    for text, label in zip(content, labels):
        content_id.append([words_id[i] for i in text if i in words_id])
        label_id.append(category_id[label])
    
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(content_id, MAX_LEN)
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(category_id))
    return x_pad, y_pad
    

def batch_iter(x, y):
    '''
        为神经网络的训练准备经过shuffle的批次的数据
    '''
    num_batch = int((len(x) - 1) / BATCH_SIZE) + 1
    indices = np.random.permutation(np.arange(len(x)))
    
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * BATCH_SIZE
        end_id = min((i + 1) * BATCH_SIZE, len(x))
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]   
    

train = read_file('train')
# 查看label类别
print(train['label'].drop_duplicates())
words = build_vocab(train)
words_id = read_vocab(words)
category_id = read_category(train)
x_pad, y_pad = preocess_file(train, words_id, category_id)
batch_iter(x_pad, y_pad)
test = read_file('test')
val = read_file('val')

