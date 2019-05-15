# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:49:10 2019

@author: pc
"""

import pandas as pd
import jieba
from collections import Counter

TRAIN_PATH = 'E:/task2/cnews.train.txt'
STOPWORDS_PATH = 'E:/task2/ChineseStopWords.txt'
VOCAB_SIZE = 5000

def read_file(file_name):
    '''
        读文件
    '''
    file_path = {'train': TRAIN_PATH}
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


def get_stopwordslist(path):
    stopwords = [line.strip() for line in open(path, 'r', encoding='utf-8').readlines()]  
    return stopwords          


def pre_data(data):
    content = []
    stop_words = get_stopwordslist(STOPWORDS_PATH)
    for text in data['text']:
        for uchar in text:
            # 判断是否为汉字
            if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                continue
             # 判断是否为数字
            if uchar >= u'\u0030' and uchar<=u'\u0039':    
                continue
            # 判断是否为英文字母
            if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):     
                continue
            else:
                text = text.replace(uchar, '')
        # jieba分词
        text_jieba = jieba.cut(text, cut_all=False)
        # 去停用词
        text = []
        for word in text_jieba:
            if word not in stop_words:
                text.append(word)
        content.append(text)
    
    return content

def get_wordsCounter(data):
    '''
        词，字符频率统计
    '''
    all_content = []
    # 把所有的text放到一个list中
    for content in data:
        all_content.extend(content)
    # 对字符频率统计
    counter = Counter(all_content)
    count_pairs = counter.most_common(VOCAB_SIZE - 1)
    
    words_counter = pd.DataFrame([i[0] for i in count_pairs], columns={'words'})
    words_counter['counter'] = [i[1] for i in count_pairs]
    return words_counter
        

train = read_file('train')
train = train.iloc[:100]
content = pre_data(train)
counter_words = get_wordsCounter(content)