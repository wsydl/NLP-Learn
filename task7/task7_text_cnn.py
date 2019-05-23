# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:06:27 2019

@author: pc
"""
from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Dense, Flatten
from keras.engine.input_layer import Input
from keras.layers.merge import concatenate
from keras.models import load_model

from keras.utils import plot_model
#import os
#os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


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

def model_CNN(X_train, X_test, y_train, y_test, index_word, embedding_matrix):
    '''
        模型结构：
            嵌入层：将词编码数据转换为固定尺寸的稠密向量，同时把词向量矩阵加载到Embedding层
            卷积池化层：256 * 3 * 3
            卷积池化层：128 * 3 * 3
            Dropout：0.1
            BatchNormalization: 批量标准化层，在每一个批次的数据中标准化前一层的激活项
            全连接：256,'relu'
            分类器：2， 'softmax'
    '''
    model = Sequential()
    model.add(Embedding(len(index_word) + 1,             # imput_dim: 词汇表大小，即最大整数index+1
                        FASTEXT_SIZE,                    # output_dim: 词向量的维度
                        weights=[embedding_matrix],      # 加载词向量矩阵
                        input_length=LEN_WORDS,           # input_lenth: 输入序列的长度
                        trainable=False))                # 设置trainable=False使得这个编码层不可再训练
    # filters:输出空间的维度，kernel_size: 1D 卷积窗口的长度，padding:"same" 表示填充输入以使输出具有与原始输入相同的长度
    model.add(Conv1D(256, 3, padding='same'))   
    # pool_size:最大池化的窗口大小, strides:作为缩小比例的因数
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(128, 3, padding='same'))
    model.add(MaxPooling1D(3, 3, padding='same'))
    model.add(Conv1D(64, 3, padding='same'))
    
    model.add(Flatten())
    # rate: 在 0 和 1 之间浮动。需要丢弃的输入比例
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    # units: 正整数，输出空间维度
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
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
    model.save('E:/task7/cnn_model.h5')
    #生成一个模型图，第一个参数为模型，第二个参数为要生成图片的路径及文件名，还可以指定两个参数：
    #show_shapes:指定是否显示输出数据的形状，默认为False
    #show_layer_names:指定是否显示层名称，默认为True
    plot_model(model,to_file='E:/task7/cnn.png',show_shapes=True,show_layer_names=False)




def model_TextCNN(X_train, X_test, y_train, y_test, index_word, embedding_matrix):
    '''
        模型结构：
            词嵌入，
            卷积池化 * 3：256 * 3 * 4
            拼接三个模型的输出向量，
            全连接，
            Dropout，
            全连接
    '''
    # shape: 一个尺寸元组（整数）表明期望的输入是按批次的LEN_WORDS维向量
    main_input = Input(shape=(LEN_WORDS, ), dtype='float32')
    embed = Embedding(len(index_word) + 1,             
                        FASTEXT_SIZE,                    
                        weights=[embedding_matrix],      
                        input_length=LEN_WORDS,           
                        trainable=False)(main_input)
    # 词窗大小分别为3，4，5
    # strides指明卷积的步长
    cnn1 =  Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=4)(cnn1)
    cnn2 =  Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=4)(cnn2)
    cnn3 =  Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=4)(cnn3)  
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)                 
    flat = Flatten()(cnn)
    drop = Dropout(0.1)(flat)
    main_output = Dense(2, activation='softmax')(drop)
    model = Model(inputs=main_input, output=main_output)
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
    plot_model(model, to_file='E:/task7/textcnn.png', show_shapes=True, show_layer_names=False)
    model.save('E:/task7/textcnn_model.h5')
                                

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
# 构建CNN分类模型
model_CNN(X_train, X_test, y_train_label, y_test_label, index_word, wordEmbedding)
model_TextCNN(X_train, X_test, y_train_label, y_test_label, index_word, wordEmbedding)


