from torch.nn import Module
from xml.etree import ElementTree as et
import os
from typing import List
import re
from collections import deque


def clean_data(text: str) -> str:
    '''
    [a-zA-Z0-9] 字母数字
    [\u4e00-\u9fa5] 汉字的utf-8 code范围
    '''

    # 保留字母、数字、汉字和标点符号(),.!?":
    def remove_others(s):
        s = re.sub(r'(),.!?":', ' ', s)
        return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', ' ', s)

    # 删除多余的空白(including spaces, tabs, line breaks)'''
    def remove_whitespaces(s):
        return re.sub(r'\s{2,}', ' ', s)

    def remove_atpeople(s):
        '''删除文本中@与其后面第一个空格之间的内容'''
        s = re.sub(r'@', ' @', s)
        s = re.sub(r':', ': ', s)
        ls = s.split()
        nls = []
        for t in ls:
            if t[0] == '@':
                continue
            else:
                nls.append(t)

        return ' '.join(nls)

    return remove_atpeople(remove_whitespaces(remove_others(text)))

class DataSource(Module):

    def __init__(self, file_path: {}):
        super(DataSource, self).__init__()
        self.text_filepath = file_path["text_filepath"]
        self.text_filename = file_path["text_filename"]
        self.label_filepath = file_path["label_filepath"]
        self.label_filename = file_path["label_filename"]

        self.label_word_to_indexer = {} #key=词  wordto index
        self.label_indexer_to_word = {} #key=索引  index to word
        self.text_word_to_indexer = {}
        self.text_indexer_to_word = {}

    def build_vacab(self, data: List, label: bool = 0):
        """

        :param data:经过paser出来的数据
        :param label: data是否是label数据
        :return:建立vocab
        """
        data_ = deque()
        for text in data:
            data_.extend(text.split(' ')) # 一句话 把训练集中全部词拿出来
        data_ = list(set(data_)) #训练集en 中全部单词

        if label == 0:   # 表示data_en
            for i, token in enumerate(data_):
                self.text_word_to_indexer[token] = i
                self.text_indexer_to_word[i] = token
        elif label == 1:
            for i, token in enumerate(data_):
                self.label_word_to_indexer[token] = i
                self.label_indexer_to_word[i] = token


    def forward(self, paser):
        """

        :param paser: 文本解析器，返回deque(str, ..., str)，此外text和label使用的是一个解析器
        :return:
        """
        # print(paser) 函数
        data_en = deque()
        data_de = deque()
        data_Source = {"text": deque(), "label": deque()}

        for i, file_name in enumerate(self.text_filename):
            # print((i,file_name)) #(0, 'train.tags.de-en.en')  只调用一次
            # print(os.path.join(self.text_filepath, file_name))   #./iwslt2016/de-en\train.tags.de-en.en
            # print(paser(os.path.join(self.text_filepath, file_name))) # 数据中全部的句子
            data_en.extend(paser(os.path.join(self.text_filepath, file_name)))
            # print(data_en[1])
            # print(len(data_en)) #196884
            data_de.extend(paser(os.path.join(self.label_filepath, self.label_filename[i])))
            # print(len(data_de)) #196884
        #     ---->
        # data_en.extend(paser(os.path.join(self.text_filepath,   self.text_filename[0])))
        # data_de.extend(paser(os.path.join(self.label_filepath, self.label_filename[0])))
        self.build_vacab(data_en, 0)
        self.build_vacab(data_de, 1)
        for en, de in zip(data_en, data_de):
            en_list = en.split(' ')#空格切分
            de_list = de.split(' ')
            de_numerized = deque()
            en_numerized = deque()
            for en_token, de_token in zip(en_list, de_list):
                de_numerized.append(self.label_word_to_indexer[de_token])
                en_numerized.append(self.text_word_to_indexer[en_token])
            data_Source["text"].append(en_numerized)
            data_Source["label"].append(de_numerized)

        data_Source["text_word_to_indexer"] = self.text_word_to_indexer
        data_Source["label_word_to_indexer"] = self.label_word_to_indexer
        data_Source["text_indexer_to_word"] = self.text_indexer_to_word
        data_Source["label_indexer_to_word"] = self.label_indexer_to_word

        return data_Source

def xml_paser(file_name):
    """

    :param file_name:
    :return: 经过文本清理后的List，每个元素对应一句话
    """
    root = et.parse(file_name).getroot()[0]
    data = deque()
    for doc in root:
        for tile in doc:
            if tile.tag == "seg":
                data.append(clean_data(tile.text))
    return data

def text_paser(file_name):

    with open(file_name, encoding= 'utf-8') as f:
        file = f.readlines()
    data = deque()
    for line in file:
        if line[0] != '<':
            data.append(line)
    return data

if __name__ == '__main__':

    test_data_path = {"text_filepath": "./iwslt2016/de-en",
                 "text_filename": ["IWSLT16.TED.dev2010.de-en.en.xml"],  #列表值
                 "label_filepath": "./iwslt2016/de-en",
                 "label_filename": ["IWSLT16.TED.dev2010.de-en.de.xml"]}

    train_data_path = {"text_filepath": "./iwslt2016/de-en",
                      "text_filename": ["train.tags.de-en.en"],
                      "label_filepath": "./iwslt2016/de-en",
                      "label_filename": ["train.tags.de-en.de"]}
    # print(len(train_data_path))  4
    testdatasource = DataSource(train_data_path)
    data = testdatasource(text_paser)

    print(data["text"][0])
    print(data["label"][0])
