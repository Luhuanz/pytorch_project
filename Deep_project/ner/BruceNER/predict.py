#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博

import time
import tensorflow as tf
from importlib import import_module
import argparse
import tensorflow_addons as tf_ad
import json
import numpy as np

# 自定义
import data_utils
from data_loader import load_model_dataset


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model',default="Bilstm", type=str, help='choose a model: Bilstm')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'BruceData'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()
    print("Loading data...")

    train_data, dev_data, test_data, train_sentences, test_sentences, dev_sentences, word_to_id, id_to_word, tag_to_id, id_to_tag = load_model_dataset(config)

    embedding_pretrained = data_utils.load_word2vec(config, id_to_word)

    config.n_vocab = len(word_to_id)


    time_dif = data_utils.get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.MyModel(config,embedding_pretrained)

    optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(config.save_path))

    while True:
        text = input("input:")
        dataset = tf.keras.preprocessing.sequence.pad_sequences([[word_to_id.get(char, 0) for char in text]],
                                                                padding='post')
        print(dataset)
        logits = model.predict(dataset)

        for index in logits:
            viterbi_path = np.argmax(index, axis=-1)
        print(viterbi_path)
        print([id_to_tag[id] for id in viterbi_path])

        entities_result = data_utils.format_result(list(text), [id_to_tag[id] for id in viterbi_path])
        print(json.dumps(entities_result, indent=4, ensure_ascii=False))
