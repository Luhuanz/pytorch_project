#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce

import jieba
import math
import random
import codecs
import numpy as np
import os
import time
from datetime import timedelta

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def check_bio(tags):
    """
    检测输入的tags是否是bio编码
    如果不是bio编码
    那么错误的类型
    (1)编码不在BIO中
    (2)第一个编码是I
    (3)当前编码不是B,前一个编码不是O
    :param tags:
    :return:
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        tag_list = tag.split("-")
        if len(tag_list) != 2 or tag_list[0] not in set(['B','I']):
            #非法编码
            return False
        if tag_list[0] == 'B':
            continue
        elif i == 0 or tags[i-1] == 'O':
            #如果第一个位置不是B或者当前编码不是B并且前一个编码0，则全部转换成B
            tags[i] = 'B' + tag[1:]
        elif tags[i-1][1:] == tag[1:]:
            # 如果当前编码的后面类型编码与tags中的前一个编码中后面类型编码相同则跳过
            continue
        else:
            # 如果编码类型不一致，则重新从B开始编码
            tags[i] = 'B' + tag[1:]
    return True

def bio_to_bioes(tags):
    """
    把bio编码转换成bioes编码
    返回新的tags
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            # 直接保留，不变化
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            # 如果tag是以B开头，那么我们就要做下面的判断
            # 首先，如果当前tag不是最后一个，并且紧跟着的后一个是I
            if (i+1) < len(tags) and tags[i+1].split('-')[0] == 'I':
                # 直接保留
                new_tags.append(tag)
            else:
                # 如果是最后一个或者紧跟着的后一个不是I，那么表示单子，需要把B换成S表示单字
                new_tags.append(tag.replace('B-','S-'))
        elif tag.split('-')[0] == 'I':
            # 如果tag是以I开头，那么我们需要进行下面的判断
            # 首先，如果当前tag不是最后一个，并且紧跟着的一个是I
            if (i+1) < len(tags) and tags[i+1].split('-')[0] == 'I':
                # 直接保留
                new_tags.append(tag)
            else:
                # 如果是最后一个，或者后一个不是I开头的，那么就表示一个词的结尾，就把I换成E表示一个词结尾
                new_tags.append(tag.replace('I-', 'E-'))

        else:
            raise Exception('非法编码')
    return new_tags

def bioes_to_bio(tags):
    """
    BIOES->BIO
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == "B":
            new_tags.append(tag)
        elif tag.split('-')[0] == "I":
            new_tags.append(tag)
        elif tag.split('-')[0] == "S":
            new_tags.append(tag.replace('S-','B-'))
        elif tag.split('-')[0] == "E":
            new_tags.append(tag.replace('E-','I-'))
        elif tag.split('-')[0] == "O":
            new_tags.append(tag)
        else:
            raise Exception('非法编码格式')
    return new_tags


def create_dico(item_list):
    """
    对于item_list中的每一个items，统计items中item在item_list中的次数
    item:出现的次数
    :param item_list:
    :return:
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    创建item to id, id_to_item
    item的排序按词典中出现的次数
    :param dico:
    :return:
    """
    sorted_items = sorted(dico.items(), key=lambda x:(-x[1],x[0]))
    id_to_item = {i:v[0] for i,v in enumerate(sorted_items)}
    item_to_id = {v:k  for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def load_word2vec(config, id_to_word):
    """
    :param config:
    :param id_to_word:
    :param word_dim:
    :return:
    """
    if os.path.exists(config.embedding_matrix_file):
        embedding_mat = np.load(config.embedding_matrix_file)
        return embedding_mat
    else:
        pre_trained = {}
        emb_invalid = 0
        for i, line in enumerate(codecs.open(config.emb_file, 'r', encoding='utf-8')):
            line = line.rstrip().split()
            if len(line) == config.embsize + 1:
                pre_trained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid = emb_invalid + 1

        if emb_invalid > 0:
            print('waring: %i invalid lines' % emb_invalid)

        num_words = len(id_to_word)
        embedding_mat = np.zeros([num_words, config.embsize])
        for i in range(num_words):
            word = id_to_word[i]
            if word in pre_trained:
                embedding_mat[i] = pre_trained[word]
            else:
                pass
        print('加载了 %i 个字向量' % len(pre_trained))
        np.save(config.embedding_matrix_file, embedding_mat)
        return embedding_mat

def augment_with_pretrained(dico_train, emb_path, test_words):
    """
    :param dico_train:
    :param emb_path:
    :param test_words:
    :return:
    """
    assert os.path.isfile(emb_path)

    #加载与训练的词向量
    pretrained = set(
        [
            line.rsplit()[0].strip() for line in codecs.open(emb_path, 'r', encoding='utf-8')
        ]
    )


    if test_words is None:
        for word in pretrained:
            if word not in dico_train:
                dico_train[word] = 0
    else:
        for word in test_words:
            if any(x in pretrained for x in
                   [word, word.lower()]
                   ) and word not in dico_train:
                dico_train[word] = 0

    word_to_id, id_to_word = create_mapping(dico_train)

    return dico_train, word_to_id, id_to_word

def get_X_and_Y_data(dataset, max_len, num_classes):
    x_data = [data[1] for data in dataset]
    x_data = pad_sequences(x_data, maxlen=max_len, dtype='int32',padding='post', truncating='post', value=0)
    y_data = [data[2] for data in dataset]
    y_data = pad_sequences(y_data, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)
    #y_data = to_categorical(y_data, num_classes=num_classes)
    return x_data, y_data

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}

def check_label(front_label,follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False

def format_result(chars, tags):
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result


