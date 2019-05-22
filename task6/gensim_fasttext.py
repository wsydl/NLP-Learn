# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:08:29 2019

@author: pc
"""

from gensim.models.fasttext import FastText
from keras.preprocessing.text import Tokenizer
import numpy as np

FASTEXT_SIZE = 100


def read_file(path):
    with open(path, 'r', encoding="UTF-8") as f:
        data = []
        labels = []
        for line in f:
            data.append(line.split('\t')[0])
            labels.append(line.split('\t')[1])
    return data, labels


def get_tokenizer(data):
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(data)
    # 对应的单词和数字的映射关系
    word_index = tokenizer.word_index 

    return word_index
    


def get_fasttext_voc(data, word_index):
    '''
        利用fasttext获取词向量
    '''
    fasttext_model = FastText([data], 
                              size=FASTEXT_SIZE,         # 需要学习的嵌入大小(默认为100)
                              window=3,         # 上下文窗口大小(默认5)
                              min_count=1,      # 忽略出现次数低于此值的单词(默认值5)
                              iter=10,          # epoch(默认5)
                              min_n = 3,        # char ngram的最小长度(默认值3)
                              max_n = 6,        # char ngram的最大长度(默认值6)
                              word_ngrams = 0)  # 如果为1，使用子单词(n-grams)信息丰富单词向量。如果是0，这就相当于Word2Vec
    # 获取词向量词典
    word_voc_dict = fasttext_model.wv.vocab
    word_voc_list = fasttext_model.wv.index2word
    # 获取词向量列表
    wordEmbedding = np.zeros((len(word_index) + 1, FASTEXT_SIZE))
    for word, i in word_index.items():
        if word in fasttext_model:
            wordEmbedding[i] = fasttext_model[word]
    
    return word_voc_dict, word_voc_list, wordEmbedding

# 读文件
data, _ = read_file('E:/task6/merge.txt')
# 获取单词-数字的映射关系
word_index = get_tokenizer(data)

# fasttext获得词向量
word_voc_dict, word_voc_list, wordEmbedding = get_fasttext_voc(data, word_index)