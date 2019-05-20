# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:00:03 2019

@author: pc
"""

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def pre_news(data):
    '''
        数据预处理：
            划分训练集，测试集
            文本特征向量化
    '''
    # 随机采样20%作为测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                        data.target,
                                                        test_size = 0.2,
                                                        random_state=33)
    # 文本特征向量化
    vec = CountVectorizer()
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def train_model_bayes(data, X_train, y_train, X_test, y_test):
    '''
        使用朴素贝叶斯进行训练并预测
    '''
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    
    # 获取结果报告
    print('acc:', clf.score(X_test, y_test))
    print(classification_report(y_test, y_pre, target_names=data.target_names))
    return y_pre
    

# 获取数据
news = fetch_20newsgroups(subset='all')
# 输出数据长度
print('len(nes):', len(news.data))
# 查看新闻类别
pprint(list(news.target_names))

# 数据预处理
X_train, X_test, y_train, y_test = pre_news(news)

# 使用朴素贝叶斯进行训练
y_pre = train_model_bayes(news, X_train, y_train, X_test, y_test)