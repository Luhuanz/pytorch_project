#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce

import codecs
import data_utils
from tqdm import tqdm
import pickle
import os
import itertools

def load_sentences(path):
    """
    加载数据集，每一行至少包含一个汉字和一个标记
    句子和句子之间是以空格进行分割
    最后返回句子集合
    :param path:
    :return:
    """
    # 存放数据集
    sentences = []
    # 临时存放每一个句子
    sentence = []
    for line in tqdm(codecs.open(path, 'r', encoding='utf-8')):
        # 去掉两边空格
        line = line.strip()
        # 首先判断是不是空，如果是则表示句子和句子之间的分割点
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                # 清空sentence表示一句话完结
                sentence = []
        else:
            if line[0] == " ":
                continue
            else:
                word = line.split()
                assert  len(word) >= 2
                sentence.append(word)
    # 循环走完，要判断一下，防止最后一个句子没有进入到句子集合中
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences, tag_scheme):
    """
    更新为指定编码
    :param sentences:
    :param tag_scheme:
    :return:
    """
    for i, s in tqdm(enumerate(sentences)):
        tags = [w[-1] for w in s]
        if not data_utils.check_bio(tags):
            s_str = "\n".join(" ".join(w) for w in s)
            raise Exception("输入的句子应为BIO编码，请检查输入句子%i:\n%s" % (i, s_str))

        if tag_scheme == "BIO":
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag

        if tag_scheme == "BIOES":
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception("非法目标编码")

def word_mapping(sentences):
    """
    构建字典
    :param sentences:
    :return:
    """
    word_list = [[x[0] for x in s] for s in sentences]
    dico = data_utils.create_dico(word_list)
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = data_utils.create_mapping(dico)
    return dico, word_to_id, id_to_word

def tag_mapping(sentences):
    """
    构建标签字典
    :param sentences:
    :return:
    """
    tag_list = [[x[1] for x in s] for s in sentences]
    dico = data_utils.create_dico(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(dico)
    return dico, tag_to_id, id_to_tag

def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理，返回list其实包含
    -word_list
    -word_id_list
    -word char indexs
    -tag_id_list
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param train:
    :return:
    """
    none_index = tag_to_id['O']

    data = []
    for s in sentences:
        word_list = [ w[0] for w in s]
        word_id_list = [word_to_id[w if w in word_to_id else '<UNK>'] for w in word_list]
        if train:
            tag_id_list = [tag_to_id[w[-1]] for w in s]
        else:
            tag_id_list = [none_index for w in s]
        data.append([word_list, word_id_list, tag_id_list])

    return data

def load_model_dataset(config):

    if os.path.exists(config.datasetpkl):
        datasetpkl = pickle.load(open(config.datasetpkl, 'rb'))
        train_sentences = datasetpkl['train']
        dev_sentences = datasetpkl['dev']
        test_sentences = datasetpkl['test']
    else:
        # 加载数据集
        train_sentences = load_sentences(config.train_path)
        dev_sentences = load_sentences(config.dev_path)
        test_sentences = load_sentences(config.test_path)

        # 转换编码 bio转bioes
        update_tag_scheme(train_sentences, config.tag_schema)
        update_tag_scheme(test_sentences, config.tag_schema)
        update_tag_scheme(dev_sentences, config.tag_schema)
        datasetpkl = {}
        datasetpkl['train'] = train_sentences
        datasetpkl['dev'] = dev_sentences
        datasetpkl['test'] = test_sentences
        pickle.dump(datasetpkl, open(config.datasetpkl, 'wb'))

    # 创建单词映射及标签映射
    if not os.path.isfile(config.map_file):
        if config.pre_emb:
            dico_words_train = word_mapping(train_sentences)[0]
            dico_word, word_to_id, id_to_word = data_utils.augment_with_pretrained(
                dico_words_train.copy(),
                config.emb_file,
                list(
                    itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in test_sentences]
                    )
                )
            )
        else:
            _, word_to_id, id_to_word = word_mapping(train_sentences)

        _, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        with open(config.map_file, "wb") as f:
            pickle.dump([word_to_id, id_to_word, tag_to_id, id_to_tag], f)
    else:
        with open(config.map_file, 'rb') as f:
            word_to_id, id_to_word, tag_to_id, id_to_tag = pickle.load(f)

    if os.path.exists(config.modeldatasetpkl):
        modeldatasetpkl = pickle.load(open(config.modeldatasetpkl, 'rb'))
        train_data = modeldatasetpkl['train']
        dev_data = modeldatasetpkl['dev']
        test_data = modeldatasetpkl['test']
    else:
        train_data = prepare_dataset(
            train_sentences, word_to_id, tag_to_id
        )

        dev_data = prepare_dataset(
            dev_sentences, word_to_id, tag_to_id
        )

        test_data = prepare_dataset(
            test_sentences, word_to_id, tag_to_id
        )
        modeldatasetpkl = {}
        modeldatasetpkl['train'] = train_data
        modeldatasetpkl['dev'] = dev_data
        modeldatasetpkl['test'] = test_data
        pickle.dump(modeldatasetpkl, open(config.modeldatasetpkl, 'wb'))

    return train_data, dev_data, test_data, train_sentences, test_sentences, dev_sentences, word_to_id, id_to_word, tag_to_id, id_to_tag