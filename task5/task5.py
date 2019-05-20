# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:53:50 2019

@author: pc
"""


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def correct_rate(predict_labels, y):
    # 查看预测对了几个
    n = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == y[i]):
            n = n + 1
    print('高斯贝叶斯：')
    # 输出正确率
    print(n/499.0)
    
    return n
    

# 高斯贝叶斯
def train_model_GaussianNB():
    pass
    clf3 = GaussianNB()
    # 训练模型
    clf3.fit(X[499:], y[499:])
    predict_labels = clf3.predict(X[0:499])
    n = correct_rate(predict_labels, y)
    # 混淆矩阵
    confusion_matrix(y[0:499], predict_labels)


from sklearn.naive_bayes import MultinomialNB

# 多项式贝叶斯
def train_model_MultinomialNB():
    pass
    clf = MultinomialNB()
    # 训练模型
    clf.fit(X[499:], y[499:])
    # 预测训练集
    predict_labels = clf.predict(X[0:499])
    # 查看预测对数目
    n = correct_rate(predict_labels, y)
    confusion_matrix(y[0:499], predict_labels)
    
from sklearn.naive_bayes import BernoulliNB    
    
# 伯努利模型
def train_model_BernoulliNB():
    pass
    clf2 = BernoulliNB()
    clf2.fit(X[499:], y[499:])
    predict_labels = clf2.predict(X[0:499])
    n = correct_rate(predict_labels, y)
    # 混淆矩阵
    confusion_matrix(y[0:499], predict_labels)
    
