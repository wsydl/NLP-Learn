# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:01:50 2019

@author: pc
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np

# 查看tensorflow版本
print(tf.__version__)

# 下载imdb数据集
imdb = keras.datasets.imdb
# 参数num_words=10000保留训练数据中出现频率最高的10,000个单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 探索数据
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
# 每篇文本长度不同
print(len(train_data[0]), len(train_data[1]))