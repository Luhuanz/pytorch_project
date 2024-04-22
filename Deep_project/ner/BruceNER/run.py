#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time
import tensorflow as tf
from importlib import import_module
import argparse


# 自定义
import data_utils
from data_loader import load_model_dataset


parser = argparse.ArgumentParser(description='Chinese NER')
parser.add_argument('--model',default="Bilstm", type=str, help='choose a model: Bilstm')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'BruceData'  # 数据集

    # 随机初始化:random
    # embedding = 'embedding_SougouNews.npz'
    # if args.embedding == 'random':
    #     embedding = 'random'
    model_name = args.model  # 'Bilstm'  # Bilstm

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()
    print("Loading data...")

    train_data, dev_data, test_data, train_sentences, test_sentences, dev_sentences, word_to_id, id_to_word, tag_to_id, id_to_tag = load_model_dataset(config)


    time_dif = data_utils.get_time_dif(start_time)
    print("Time usage:", time_dif)

    embedding_pretrained = data_utils.load_word2vec(config, id_to_word)

    train_X, train_Y = data_utils.get_X_and_Y_data(train_data, config.max_len, len(tag_to_id))

    dev_X, dev_Y = data_utils.get_X_and_Y_data(dev_data, config.max_len, len(tag_to_id))

    test_X, test_Y = data_utils.get_X_and_Y_data(test_data, config.max_len, len(tag_to_id))

    # train
    model = x.MyModel(config, embedding_pretrained)

    model.build(input_shape=(None, config.max_len))

    #model = model.createModel(input=(config.max_len,))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=config.save_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]

    history = model.fit(
        x=train_X,
        y= train_Y,
        validation_data=(dev_X, dev_Y),
        batch_size=512,
        epochs=20,
        callbacks = callbacks
    )

