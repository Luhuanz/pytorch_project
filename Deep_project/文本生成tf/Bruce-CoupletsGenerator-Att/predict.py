#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
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
    max_length_targ, max_length_input = max_length(target_tensor), max_length(input_tensor)

    # 采用 80 - 20 的比例切分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    # 显示长度
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    BUFFER_SIZE = len(input_tensor_train)

    config.steps_per_epoch = len(input_tensor_train) // config.batch_size

    vocab_input_size = len(input_tokenizer.word_index) + 1
    vocab_targ_size = len(targ_tokenizer.word_index) + 1
    config.num_encoder_tokens = vocab_input_size
    config.num_decoder_tokens = vocab_targ_size

    #第一步: 准备要加载的numpy数据
    #第二步: 使用 tf.data.Dataset.from_tensor_slices()函数进行加载
    #第三步: 使用shuffle()打乱数据
    #第四步: 使用map()函数进行预处理
    #第五步: 使用batch()函数设置batchsize值
    #第六步: 根据需要使用repeat()设置是否循环迭代数据集


    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)

    model = x.MyModel(config)
    encoder, decoder = model.createModel()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    def evaluate(sentence):
        attention_plot = np.zeros((max_length_targ, max_length_input))

        sentence = preprocess_sentence(sentence)

        inputs = [input_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=max_length_input,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, 256))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # 存储注意力权重以便后面制图
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += targ_tokenizer.index_word[predicted_id] + ' '

            if targ_tokenizer.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # 预测的 ID 被输送回模型
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot


    # 注意力权重制图函数
    def plot_attention(attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()


    def predict(sentence):
        result, sentence, attention_plot = evaluate(sentence)

        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))


    checkpoint.restore(tf.train.latest_checkpoint(config.save_path))

    predict(u'晚 风 摇 树 树 还 挺')
