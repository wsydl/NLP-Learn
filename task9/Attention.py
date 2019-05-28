# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:16:26 2019

@author: pc
"""
import keras.backend as K
from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        '''
            初始化一些需要的参数
        '''
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)
       
        
    def build(self, input_shape):
        '''
            具体定义权值是怎么样的
            W: (EMBED_SIZE, ATTENTION_SIZE)
            b: (ATTENTION_SIZE, 1)
            u: (ATTENTION_SIZE, 1)
        '''
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)
        
        
    def call(self, x, mask=None):
        '''
            定义向量是如何进行运算的
            input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
            et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        '''
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        '''
            定义该层输出的大小
        '''
        return (input_shape[0], input_shape[-1])