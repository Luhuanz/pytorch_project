#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout
import tensorflow_addons as tf_ad


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'IDCNNCRF'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name  # 日志存储
        self.map_file = dataset + '/data/map.pkl'  # 字典映射文件
        self.emb_file = dataset + '/data/wiki_100.utf8'  # 外部词向量文件
        self.datasetpkl = dataset + '/data/dataset.pkl'  # 数据存储文件
        self.modeldatasetpkl = dataset + '/data/modeldataset.pkl'  # 模型需要数据文件
        self.embedding_matrix_file = dataset + '/data/word_embedding_matrix.npy'  # 词向量压缩好的文件
        self.embsize = 100  # 词向量维度
        self.pre_emb = True  # 是否需要词嵌入
        self.tags_num = 13  # 标签数量

        self.dropout = 0.5  # 随机失活
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 100  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.max_len = 200  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.tag_schema = "BIOES"  # 编码数量

class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(config.n_vocab, config.embsize)
        self.conv1 = Conv1D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   dilation_rate=1)
        self.conv2 = Conv1D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   dilation_rate=1)
        self.conv3 = Conv1D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   dilation_rate=2)

        self.dense = Dense(config.tags_num)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(config.tags_num, config.tags_num)),
                                             trainable=False)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, text,labels=None,training=None):
        # print(text.shape) #(128, 200)
        x = tf.math.not_equal(text, 0)
        x = tf.cast(x, dtype=tf.int32)
        text_lens = tf.math.reduce_sum(x, axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
