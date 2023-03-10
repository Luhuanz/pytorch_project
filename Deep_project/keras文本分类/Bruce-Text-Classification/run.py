#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
from importlib import import_module
from utils import build_dataset, get_time_dif, build_net_data
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model',default="TextCNN2D", type=str, help='choose a model: TextCNN1D, TextCNN2D,TextRNN')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
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

    train_x, train_y = build_net_data(train_data, config)
    dev_x, dev_y = build_net_data(dev_data, config)
    test_x, test_y = build_net_data(test_data, config)
    # train
    config.n_vocab = len(vocab)
    model = x.MyModel(config)

    model.build(input_shape=(None, config.max_len))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=config.save_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(dev_x, dev_y ),
        batch_size=512,
        epochs=20,
        callbacks=callbacks
    )

