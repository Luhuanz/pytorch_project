#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Dropout, Flatten, MaxPooling1D, Input,concatenate
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = tf.convert_to_tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量

        self.dropout = 0.5                                              # 随机失活
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.shape[1]\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''

class CnnModel(tf.keras.Model):
    def __init__(self, config):
        super(CnnModel, self).__init__()
        self.config = config


    def createModel(self, input):
        main_input = Input(shape=input, dtype='float64')
        if self.config.embedding_pretrained is not None:
            self.embedding = Embedding(input_dim=self.config.embedding_pretrained.shape[0], output_dim=self.config.embedding_pretrained.shape[1],
                                       input_length=self.config.max_len,weights=[self.config.embedding_pretrained],trainable=False)
        else:
            self.embedding = Embedding(self.config.n_vocab, self.config.embed, input_length=self.config.max_len)

        # embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
        embed = self.embedding(main_input)
        # 词窗大小分别为3,4,5
        cnn1 = Conv1D(self.config.num_filters, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=2)(cnn1)
        cnn2 = Conv1D(self.config.num_filters, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=2)(cnn2)
        cnn3 = Conv1D(self.config.num_filters, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=2)(cnn3)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(self.config.dropout)(flat)
        main_output = Dense(self.config.num_classes, activation='softmax')(drop)
        model = tf.keras.Model(inputs=main_input, outputs=main_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
