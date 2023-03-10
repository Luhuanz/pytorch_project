#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博   tf no  keras

import time
import tensorflow as tf
from importlib import import_module
import argparse
import tensorflow_addons as tf_ad


# 自定义
import data_utils
from data_loader import load_model_dataset


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model',default="IDCNNCRF", type=str, help='choose a model: BilstmCRF')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':


    if tf.test.is_gpu_available():
        print("GPU is available!")
    else:
        print("GPU is not available.")

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

    config.n_vocab = len(word_to_id)


    time_dif = data_utils.get_time_dif(start_time)
    print("Time usage:", time_dif)

    embedding_pretrained = data_utils.load_word2vec(config, id_to_word)

    train_X, train_Y = data_utils.get_X_and_Y_data(train_data, config.max_len, len(tag_to_id))

    dev_X, dev_Y = data_utils.get_X_and_Y_data(dev_data, config.max_len, len(tag_to_id))

    test_X, test_Y = data_utils.get_X_and_Y_data(test_data, config.max_len, len(tag_to_id))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    train_dataset = train_dataset.shuffle(len(train_X)).batch(config.batch_size, drop_remainder=True)
# 128,200
    # train
    model = x.MyModel(config)

    optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(tf.train.latest_checkpoint(config.save_path))
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              config.save_path,
                                              checkpoint_name= config.model_name + '.ckpt',
                                              max_to_keep=3)


    def train_one_step(text_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits, text_lens


    def get_acc_one_step(logits, text_lens, labels_batch):
        paths = []
        accuracy = 0
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
            paths.append(viterbi_path)
            correct_prediction = tf.equal(
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                     dtype=tf.int32),
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                     dtype=tf.int32)
            )
            accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
        accuracy = accuracy / len(paths)
        return accuracy

best_acc = 0
step = 0
for epoch in range(config.num_epochs):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labels_batch)
            print('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            if accuracy > best_acc:
              best_acc = accuracy
              ckpt_manager.save()
              print("model saved")


print("finished")
