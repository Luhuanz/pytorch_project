#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
import tensorflow as tf
import tensorflow_addons as tf_ad


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'BilstmCRF'
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
        self.hidden_size = 128  # lstm隐藏层
        self.tag_schema = "BIOES"  # 编码数量

class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(config.n_vocab, config.embsize)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.hidden_size, return_sequences=True))
        self.dense = tf.keras.layers.Dense(config.tags_num)
#创建一个形状为(config.tags_num, config.tags_num)的张量，其中的值是随机均匀分布的，用作转移矩阵。
 # 这个转移矩阵是用于条件随机场（CRF）模型中的标签转移概率。由于这个矩阵在模型的训练过程中不需要更新，因此设置为不可训练（trainable=False）。
        self.transition_params = tf.Variable(tf.random.uniform(shape=(config.tags_num, config.tags_num)),
                                             trainable=False)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
#它接受一个文本输入和一个可选的标签输入，并返回模型的输出和长度信息。
    def call(self, text,labels=None,training=None):
#首先根据文本输入创建了一个二进制张量x，表示哪些位置是真实的输入（不是填充）。
# 然后通过将x转换为整数类型，并在最后一个维度上求和，计算了文本输入的实际长度。

        print(x)
        exit()
        x = tf.math.not_equal(text, 0)
        x = tf.cast(x, dtype=tf.int32)
        text_lens = tf.math.reduce_sum(x, axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        inputs = self.biLSTM(inputs)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
