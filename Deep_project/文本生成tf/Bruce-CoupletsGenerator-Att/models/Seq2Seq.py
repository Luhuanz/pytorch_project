#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input
from models import GRUEncoder
from models import GRUDecoder
from models import BahdanauAttention

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Seq2SeqWithAttention'
        self.train_path = dataset + '/data/train.txt'                                # 训练数据集
        self.test_path = dataset + '/data/test.txt'                                  # 测试数据集
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.num_samples = 10000                                                     # 最多10000个样本参与训练

        self.num_epochs = 200                                           # epoch数
        self.batch_size = 64                                            # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.steps_per_epoch = 0                                        #迭代多少次打印输出
        self.embedding_dim = 300
        self.hidden_size = 256                                          # 卷积核数量(channels数)
        self.num_encoder_tokens = 0                                     # encoder中token数量，在运行时赋值
        self.num_decoder_tokens = 0                                     # decoder中token数量，在运行时赋值
        self.max_encoder_seq_length = 0                                 # encoder中最大序列长度，在运行时赋值
        self.max_decoder_seq_length = 0                                 # dncoder中最大序列长度，在运行时赋值
        self.input_length = 0                                           # 输入序列长度，在运行时赋值



class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config


    def createModel(self):

        encoder = GRUEncoder.Encoder(self.config.num_encoder_tokens, self.config.embedding_dim, self.config.hidden_size, self.config.batch_size)
        decoder = GRUDecoder.Decoder(self.config.num_decoder_tokens, self.config.embedding_dim, self.config.hidden_size, self.config.batch_size)

        return encoder, decoder
