#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
# tf 2.5.0
import time
import tensorflow as tf
from importlib import import_module
from sklearn.model_selection import train_test_split
from utils import load_dataset, max_length, preprocess_sentence
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser(description='PoemsGenerator')
parser.add_argument('--model',default="Seq2Seq", type=str, help='choose a model: Seq2Seq')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'BruceCouplets'  # 数据集
    model_name = args.model  # Seq2Seq

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()
    print("Loading data...")
    input_tensor, target_tensor, input_tokenizer, targ_tokenizer = load_dataset(config.train_path, config.num_samples)

    # 计算目标张量的最大长度 （max_length）
    max_length_targ, max_length_input = max_length(target_tensor), max_length(input_tensor) #31 31

    # 采用 80 - 20 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    # 显示长度
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    BUFFER_SIZE = len(input_tensor_train) #总的数据量
#在一个 epoch需要对模型进行多少次训练更新（即更新多少个批次）
    config.steps_per_epoch = len(input_tensor_train) // config.batch_size

    vocab_input_size = len(input_tokenizer.word_index) + 1 #3784
    vocab_targ_size = len(targ_tokenizer.word_index) + 1 #3863
    config.num_encoder_tokens = vocab_input_size
    config.num_decoder_tokens = vocab_targ_size

    #第一步: 准备要加载的numpy数据
    #第二步: 使用 tf.data.Dataset.from_tensor_slices()函数进行加载
    #第三步: 使用shuffle()打乱数据
    #第四步: 使用map()函数进行预处理
    #第五步: 使用batch()函数设置batchsize值
    #第六步: 根据需要使用repeat()设置是否循环迭代数据集

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(config.batch_size, drop_remainder=True) # 64 31

    model = x.MyModel(config)
    encoder, decoder = model.createModel()

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

#用于计算模型预测结果和真实标签之间的损失函数。
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
#tf.train.Checkpoint() 是 TensorFlow 中的一个检查点对象，它可以用于保存和恢复训练过程中的模型和优化器状态。
# 在创建一个检查点对象时，我们需要将需要保存的变量传递给 tf.train.Checkpoint()，通常包括模型的参数和优化器。
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)


    #@tf.function
    def train_step(input, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(input, enc_hidden) #  每个时间段输出、 c

            dec_hidden = enc_hidden #(64, 256) #c

#在预处理阶段，我们需要将句子转换为整数序列，并在序列开头添加一个起始标记 '<start>' 和在序列结尾添加一个终止标记 '<end>'，以表示序列的开始和结束。
            dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * config.batch_size, 1) #shape=(64, 1)
# decoder第一个字符是<start>对应词表为 1
            for t in range(1, targ.shape[1]):
                # 将编码器输出 （enc_output） 传送至解码器
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output) # x c #(64, 3863) (64, 256)

                loss += loss_function(targ[:, t], predictions)
                x = targ[:, t]
                dec_input = tf.expand_dims(x, 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    for epoch in range(config.num_epochs):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (input, targ)) in enumerate(dataset.take(config.steps_per_epoch)):
            batch_loss = train_step(input, targ, enc_hidden) #enc_hidden 初始化embedding0
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=config.save_path)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / config.steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

