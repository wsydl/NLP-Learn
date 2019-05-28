# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:49:41 2019

@author: pc
"""
from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from Attention import *
from keras.utils import plot_model


FASTEXT_SIZE = 200
LEN_WORDS = 20

def read_file(path):
    with open(path, 'r', encoding="UTF-8") as f:
        data = []
        labels = []
        for line in f:
            if line.split('\t')[1] == '':
                continue
            data.append(line.split('\t')[0])
            labels.append(line.split('\t')[1])
    return data, labels


def get_tokenizer(data):
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(data)
    text_seq = tokenizer.texts_to_sequences(data)
    # 对应的单词和数字的映射关系
    word_index = tokenizer.word_index 
    index_word = tokenizer.index_word

    return word_index, index_word, text_seq


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
    # 获取词向量列表
    wordEmbedding = np.zeros((len(word_index) + 1, FASTEXT_SIZE))
    for word, i in word_index.items():
        if word in fasttext_model:
            wordEmbedding[i] = fasttext_model[word]
    
    return wordEmbedding
    

def get_label_num(data):
    data = [i.replace('\n', '') for i in data]
    y_labels = list(set(data))
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    data_labels = to_categorical([le.transform([x])[0] for x in data], num_labels)
    return data_labels

def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
        n: 要检索的样本数量
        time_steps:您的系列的时间步长
        input_dim:系列中每个元素的维数
        attention_column:链接到目标的列
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


def attention_lstm_model(X_train, X_test, y_train, y_test, index_word, embedding_matrix):
    
    inputs = Input(shape=(LEN_WORDS, ))
    embed = Embedding(len(index_word) + 1,             
                        FASTEXT_SIZE,                    
                        weights=[embedding_matrix],      
                        input_length=LEN_WORDS,           
                        trainable=False)(inputs)
    gru = Bidirectional(GRU(100, dropout=0.2, return_sequences=True))(embed)
    attention = Attention(16)(gru)
    # flat = Flatten()(attention)
    drop = Dropout(0.1)(attention)
    main_output = Dense(2, activation='softmax')(drop)
    model = Model(inputs=inputs, output=main_output)
    # 配置训练模型
    model.compile(loss='categorical_crossentropy',          # 表示目标应该是分类格式的
                  optimizer='adam',                         # 随机优化的一种方法
                  metrics=['accuracy']                      # 模型评估标准
                  )
    # 给定数量的迭代训练模型
    model.summary()
    model.fit(X_train, y_train, 
              batch_size=32, 
              epochs=15, 
              validation_data=(X_test, y_test))
    plot_model(model, to_file='E:/task9/attention_model.png', show_shapes=True, show_layer_names=False)
    model.save('E:/task9/attention_model.h5')
    

# 读文件
data, labels = read_file('E:/task6/merge.txt')

# 利用tokenizer将文字转换为数字特征
word_index, index_word, X_train_text_seq = get_tokenizer(data)
# fasttext embedding
wordEmbedding = get_fasttext_voc(data, word_index)

# 让每个文本长度相同
X_train_text_seq = pad_sequences(X_train_text_seq, maxlen=LEN_WORDS, padding='post', truncating='post')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_train_text_seq, 
                                                    labels,
                                                    test_size = 0.2,
                                                    random_state=33)
# 对类别变量编码
y_train_label = get_label_num(y_train)
y_test_label = get_label_num(y_test)
attention_lstm_model(X_train, X_test, y_train_label, y_test_label, index_word, wordEmbedding)