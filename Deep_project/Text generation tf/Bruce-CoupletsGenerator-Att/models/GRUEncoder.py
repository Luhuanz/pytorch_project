#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz): # 3784 300  hiddensize= 256  64
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz  #6
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
       # print(x) #shape=(64, 31), 64 batch  31最长每句
        x = self.embedding(x) #(64, 31, 300)
#hidden是GRU层的初始隐藏状态，它通常初始化为零向量 使用GRU层处理输入序列x，并返回所有时间步长的输出序列和最终时间步长的隐藏状态。
        output, state = self.gru(x, initial_state = hidden) #(64, 31, 256) (64, 256)
       #Encoder的输出是它内部的最后一个隐藏状态，这个向量包含了输入序列中所有的信息，并且可以被用来预测一个与之相关的输出序列。
       #LSTM只返回最后一个时间步的隐藏状态，而GRU会返回最后一个时间步的隐藏状态和每个时间步的输出。 h5
       #。输出向量是指当前时间步的隐藏状态（hidden state）与权重矩阵进行矩阵乘法和激活函数计算得到的向量。 o_i pdf上的  c

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))