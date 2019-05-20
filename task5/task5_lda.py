# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:50:45 2019

@author: pc
"""
import numpy as np
import lda
import lda.datasets
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

def load_data():
    '''
        加载lda中的数据
    '''
    X = lda.datasets.load_reuters()
    vocab = lda.datasets.load_reuters_vocab()
    titles = lda.datasets.load_reuters_titles()
    
    return X, vocab, titles


def train_model_lda(X, vocab, titles):  
    
    #指定11个主题，500次迭代
    model = lda.LDA(random_state=1, n_topics=11, n_iter=1000)
    model.fit(X)
    
    # 主题单词分布
    topic_word = model.topic_word_
    print('type(topic_word):{}'.format(type(topic_word)))
    print('shape:{}'.format(topic_word.shape))
    
    # 获取每个topic下权重最高的10个单词
    n = 10
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        print('topic {}\n -{}'.format(i, ' '.join(topic_words)).encode('utf-8'))
    
    #文档主题分布：
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))
    
    # 输入前10篇文章最可能的topic
    for n in range(20):
        topic_most_pr = doc_topic[n].argmax()
        print('doc: {} topic: {}'.format(n, topic_most_pr))
        

def train_model_lda_gensim():
    # 把文章转成list
    common_dictionary = Dictionary(common_texts)
    print(type(common_texts))
    print(common_texts[0])
    
    # 把文本转成词袋形式
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    
    # 调用lda模型，并指定10个主题
    lda = LdaModel(common_corpus, num_topics=10)
    # 检查结果
    lda.print_topic(1, topn=2)
    
#    # 用新语料库去更新
#    # 能更新全部参数
#    lda.update(other_corpus)
#    #还能单独更新主题分布， 输入为之前的参数，其中rho指学习率
#    lda.update_alpha(gammat, rho)
#    #还能单独更新词分布
#    lda.update_eta(lambdat, rho)
#    
    

# 加载数据
X, vocab, titles = load_data()
# 训练数据
train_model_lda(X, vocab, titles) 
train_model_lda_gensim()   