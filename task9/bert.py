# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:22:32 2019

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:12:55 2019

@author: pc
"""
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor

from functools import reduce
import logging
import sys
import re
import jieba
import keras
from keras.models import Sequential, Model
from keras.layers import Embedding, Masking, Dense, Bidirectional, TimeDistributed, Input, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec, FastText
import gensim
from bert_serving.client import BertClient


WORD2VECTOR_SIZE = 200
FASTEXT_SIZE = 200
LEN_WORDS = 300


def get_log(appname, lv=logging.INFO):
    logger = logging.getLogger(appname)
    formatter = logging.Formatter('%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)-4s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(console_handler)
    logger.setLevel(lv)

    return logger


def get_article(data):
    article = []
    for _, raw in data.iterrows():
        temp = (',').join([raw['title'], raw['content']])
#        punc = punctuation + u'！？｡＂ ＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.' + '\u200b, \u00A0,\u0020,\u3000,\u2002,💖,📅'
#        temp = re.sub(r"[{}]+".format(punc),'', temp)
#        temp = temp.replace('\r', '').replace('\n', '').replace('\t', '')
        for uchar in temp:
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
                temp = temp.replace(uchar, '')
        punc = u'犟砀滘鸩'
        temp = re.sub(r"[{}]+".format(punc),'', temp)
#        temp = temp.replace('犟', '').replace('砀', '').replace('滘', '').replace('鸩', '')
        article.append(temp)
        
    return article


def get_words(data):
    count = 0
    nerCorpus = []
    for i in data:
        count += 1
        logger.info(f'切分第{count}个数据')
        word = list(jieba.cut(i, cut_all=False))
        words = []
        for i in word:
            words.append(i)        
        nerCorpus.append(words)
    return nerCorpus


def get_tokenizer(data):
    tokenizer = Tokenizer(num_words=None)
    logger.info('得到文本的字典...')
    tokenizer.fit_on_texts(data)
    logger.info('将每个string的每个词转成数字...')
    sequences = tokenizer.texts_to_sequences(data)
    logger.info('对应的单词和数字的映射关系...')
    word_index = tokenizer.word_index 
    index_word = tokenizer.index_word
    
    return sequences, word_index, index_word  
    

def get_entityLabel(train, sequences, index_word):
    data_entityLabel = []
    for i, seq in enumerate(sequences):
        entityLabel = []
        entity = [x['entity'] for x in eval(train.loc[i, 'coreEntityEmotions'])]
        for index in seq:
            if index != 0 and index_word[index] in entity:
                entityLabel.append([0, 1])
            else:
                entityLabel.append([1, 0])
        data_entityLabel.append(entityLabel)
    return np.array(data_entityLabel)


def resultDeal(probs, X_test, index_word, entity_pro):
    w2v_prob = pd.read_csv('E:/nlp_souhu/output/preds_w2v_30.csv')
    for i in range(30):
        probs[str(i)] = 0.6*probs[str(i)] + 0.4*w2v_prob[str(i)]
    entity_list = []
    emotion_list = []
    print(X_test[0][0])
    for i in probs.index:
        i = int(i)
        entity = []
        prob = probs.ix[i]
        prob_dict = dict(zip(list(range(len(prob))), prob))
        for j in prob.index:
            j = int(j)
            if X_test[i][j] == 0:
                prob_dict.pop(j)
        prob = pd.Series(list(prob_dict.values()), index=list(prob_dict.keys()))
        prob.sort_values(ascending=False, inplace=True)

        n = 3
        count = 0
        for j in prob.index:
            j = int(j)
            if count >= n:
                break
            if index_word[X_test[i][j]] not in entity:
                entity.append(index_word[X_test[i][j]])
                count += 1

        entity_list.append(','.join(entity))
        emotion_list.append(','.join(['POS'] * len(entity)))

    return entity_list, emotion_list


def train_model(index_word, embedding_matrix, entityLabel, X_test, test_data):
    sequence = Input(shape=(LEN_WORDS, ), dtype='float32')
    embedding = Embedding(len(index_word) + 1, 
                          WORD2VECTOR_SIZE, 
                          weights=[embedding_matrix], 
                          input_length=LEN_WORDS, 
                          trainable=False, 
                          mask_zero=True)(sequence)
    mask = Masking(mask_value=0.)(embedding)
    blstm = Bidirectional(LSTM(256, 
                               kernel_initializer=keras.initializers.Orthogonal(seed=2019), 
                               recurrent_initializer=keras.initializers.glorot_normal(seed=2019), 
                               bias_initializer=keras.initializers.Zeros(), 
                               return_sequences=True), merge_mode='sum')(mask)
    dropout = Dropout(0.1, seed=2019)(blstm)
    blstm = Bidirectional(LSTM(128, 
                               kernel_initializer=keras.initializers.Orthogonal(seed=2019), 
                               recurrent_initializer=keras.initializers.glorot_normal(seed=2019), 
                               bias_initializer=keras.initializers.Zeros(), 
                               return_sequences=True), merge_mode='sum')(dropout)
    dropout = Dropout(0.1, seed=2019)(blstm)
    output = TimeDistributed(Dense(2, 
                                   kernel_initializer=keras.initializers.Orthogonal(seed=2019), 
                                   bias_initializer=keras.initializers.Zeros(), 
                                   activation='softmax'))(dropout)
    # 加入crf
    model = Model(sequence, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, entityLabel, batch_size=64, epochs=3)
    logger.info('输出预测结果...')
    probs = pd.DataFrame(model.predict(X_test)[:, :, 1])
    probs.to_csv('E:/nlp_souhu/output/prob.csv', index=None)
    probs = pd.read_csv('E:/nlp_souhu/output/prob.csv')



if __name__ == '__main__':
    logger = get_log('souhu_main')
    logger.info('Read train data...')
    train = pd.read_csv('E:/Task9/train.csv')
    logger.info('Read test data...')
    test = pd.read_csv('E:/Task9/test.csv')
    logger.info('Read nerDict data...')
    nerDict = pd.read_csv(NERDICT_PATH, names=['keywords'])
    for _, i in nerDict.iterrows():
        print('not cut words:', str(i['keywords']))
        jieba.suggest_freq(str(i['keywords']), True)
    train_article = get_article(train)
    test_article = get_article(test)
    logger.info('对文章进行分词...')
    train_article_words = get_words(train_article)
    test_article_words = get_words(test_article)
    all_content = train_article_words + test_article_words
    logger.info('转化文章为神经网络所用的张量...')
    sequence, word_index, index_word = get_tokenizer(all_content)

    bc = BertClient()
    bc.encode(all_content)
 
    wordEmbedding = np.zeros((len(word_index) + 1, FASTEXT_SIZE))
    for word, i in word_index.items():
        if word in model:
            wordEmbedding[i] = bc[word]
    

    X_all = pad_sequences(sequence, maxlen=LEN_WORDS, padding='post', truncating='post')
    X_train = X_all[:len(train)]
    X_test = X_all[len(train):]
    logger.info('对词语进行打标...')
    entityLabel = get_entityLabel(train, X_train, index_word)

    train_model(index_word, wordEmbedding, entityLabel, X_test, test)
    


    
    