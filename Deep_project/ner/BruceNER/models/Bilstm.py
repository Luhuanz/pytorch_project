#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Bilstm'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.vocab_path = dataset + '/data/vocab1.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict1/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name                          # 日志存储
        self.map_file = dataset + '/data/map1.pkl'                                    # 字典映射文件
        self.emb_file = dataset + '/data/wiki_100.utf8'                              # 外部词向量文件
        self.datasetpkl = dataset + '/data/dataset1.pkl'                              # 数据存储文件
        self.modeldatasetpkl = dataset + '/data/modeldataset1.pkl'                    # 模型需要数据文件
        self.embedding_matrix_file = dataset + '/data/word_embedding_matrix1.npy'     # 词向量压缩好的文件
        self.embsize = 100                                                           # 词向量维度
        self.pre_emb = True                                                          # 是否需要词嵌入
        self.tags_num = 13                                                           # 标签数量

        self.dropout = 0.5                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 200                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.hidden_size = 128                                          # lstm隐藏层
        self.tag_schema = "BIOES"                                       # 编码数量


'''Recurrent Neural Network for Text Classification'''

class MyModel(tf.keras.Model):
    def __init__(self, config, embedding_pretrained):
        super(MyModel, self).__init__()
        self.config = config
        self.embedding = Embedding(input_dim=embedding_pretrained.shape[0],
                                   output_dim=embedding_pretrained.shape[1],
                                   input_length=self.config.max_len, weights=[embedding_pretrained],
                                   trainable=True)
        self.biRNN = Bidirectional(LSTM(units=self.config.hidden_size,
                                     return_sequences=True,
                                     activation='relu',
                                     ))

        self.dropout = Dropout(self.config.dropout)
        self.out_put = Dense(self.config.tags_num, activation='softmax')
#这个函数是一个类方法，用于在 Keras 模型中构建神经网络层。build() 方法在模型第一次被调用时被自动调用，以构建模型的网络架构。
# 该方法会接收输入数据的形状 input_shape，并返回模型的输出形状。
# 在该函数中，通过调用 super() 方法来调用父类的 build() 方法，以确保模型被正确地构建。这个函数通常需要被覆盖实现，以定义模型的网络结构和参数。
    def build(self, input_shape):
        super(MyModel, self).build(input_shape)

    def call(self, x):
        # print(x.shape) #(None, 200)
        # exit()
        x = self.embedding(x) #(None, 200, 100)
        x = self.biRNN(x) #(None, 200, 256) 128x2
        x = self.dropout(x)
        x = self.out_put(x) #(None, 200, 13)

        return x