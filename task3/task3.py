# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:48:34 2019

@author: pc
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim import corpora, models
from collections import Counter
import math
from sklearn import metrics as mr


def tf_idf_weight_sklearn(words):
    '''
        使用sklearn提取文本tfidf特征
    '''
    vectorizer = CountVectorizer()      # 将词语转换成词频矩阵
    transformer = TfidfTransformer()    # 将统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(words))
    word = vectorizer.get_feature_names()   # 获取词袋模型中的所有词语
    weight = tfidf.toarray()
    weight = pd.DataFrame(weight,columns = word)
    return weight

def tf_idf_weight_gensim(words):
    '''
        使用gensim提取文本的tfidf特征
    '''
    word_list = [ sentence.split(' ') for sentence in words]
    # 赋给语料库中每个词(不重复的词)一个整数id
    dictionary = corpora.Dictionary(word_list)
    # 通过下面的方法可以看到语料库中每个词对应的id
    print(dictionary.token2id)
    new_corpus = [dictionary.doc2bow(text) for text in word_list]
    
    # 载入模型
    tfidf = models.TfidfModel(new_corpus)
    tfidf.save("my_model.tfidf")
    
    # 使用模型计算tfidf值
    tfidf = models.TfidfModel.load("my_model.tfidf")
    tfidf_vec = []
    for text in words:
        string_bow = dictionary.doc2bow(text.lower().split())
        tfidf_vec.append(tfidf[string_bow])
    
    return tfidf_vec


def get_tf(word_list, words_count):
    '''
        根据分词列表以及词频列表计算词频
    '''
    words_tf = []
    for i in range(len(word_list)):
        word_tf_dict = dict()
        for word in word_list[i]: 
            print(words_count[i][word])
            word_tf_dict[word] = words_count[i][word] / sum(words_count[i].values())
        words_tf.append(word_tf_dict)
    return words_tf


def get_contain(word, word_list):
    count = 0
    for text in word_list:
        if word in text:
            count += 1
    return count


def get_idf(word_list):
    # 统计包含该词的文档数
    all_text = []
    for text in word_list:
        all_text += text
    all_word = list(set(all_text))
    word_idf = dict()
    for word in all_word:
        word_count = get_contain(word, word_list)
        word_idf[word] = math.log(len(word_list) / (1 + word_count))
    
    return word_idf
            

def get_tfidf(words):
    '''
        手动实现TF-IDF
    '''    
    # 分词
    word_list = [sentence.split(' ') for sentence in words]
    # 统计词频
    sentence_list = []
    for sentence in word_list:
        sentence_list.append(Counter(sentence))
    # 计算tf值
    words_tf = get_tf(word_list, sentence_list)
    # 计算idf值
    words_idf = get_idf(word_list)
    # 计算TF-IDF
    tf_idf_weight = []
    for i in range(len(word_list)):
        tf_idf_dict = dict()
        for word in word_list[i]:
            tf_idf_dict[word] = words_tf[i][word] * words_idf[word]
        tf_idf_weight.append(tf_idf_dict)
    
    # 转成DataFrame
    tf_idf = pd.DataFrame()
    for word in words_idf.keys():
        value = []
        for text in tf_idf_weight:
            if word in text.keys():
                value.append(text[word])
            else:
                value.append(0.0)
        tf_idf[word] = value
        
    return tf_idf
    

corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]
tfidf_weight_list = tf_idf_weight_sklearn(corpus)
tfidf_weight_list_gensim = tf_idf_weight_gensim(corpus)
tfidf_weight = get_tfidf(corpus)

# 互信息
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
mr.adjusted_mutual_info_score(labels_true, labels_pred)