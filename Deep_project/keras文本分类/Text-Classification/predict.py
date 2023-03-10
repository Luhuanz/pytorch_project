#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博

import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from importlib import import_module
from utils import build_dataset, get_time_dif, build_net_data
import argparse


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model',default="TextCNN", type=str, help='choose a model: TextCNN, TextRNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'BruceNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset, embedding) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    test_x, test_y = build_net_data(test_data, config)
    # train
    config.n_vocab = len(vocab)
    model = x.CnnModel(config)

    model = model.createModel(input=(config.max_len,))
    model.summary()
    model.load_weights(config.save_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 评估模型
    score = model.evaluate(test_x, test_y, verbose=2)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])