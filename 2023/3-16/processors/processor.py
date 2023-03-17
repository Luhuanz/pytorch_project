import os
from loguru import logger
import torch
from tqdm import trange, tqdm
import numpy as np
import pickle
from utils.utils import write_pickle, load_pickle
from utils.utils import load_lines, write_lines
from processors.trie_tree import Trie
from processors.dataset import NERDataset
import json
from processors.vocab import Vocabulary
from os.path import join
import argparse
from transformers import BertTokenizer


class Processor(object):

    def __init__(self, config):
        self.data_path = config.data_path
        self.overwrite = config.overwrite
        self.train_file = config.train_file
        self.dev_file = config.dev_file
        self.test_file = config.test_file

    def get_input_data(self, file):
        raise NotImplemented()

    def get_train_data(self):
        logger.info('loading train data')
    #它首先根据数据类型(BertProcessor或LEBertProcessor)设置不同的文件名
        if type(self) == BertProcessor:
            file_name = 'bert_train.pkl'
        else:
            file_name = 'lebert_train.pkl'
        save_path = join(self.data_path, file_name)
        #尝试从硬盘上加载缓存的训练数据集。
#如果不存在缓存数据集或强制覆盖模式被打开，函数将调用get_input_data方法读取原始训练数据并从中提取出特征
        if self.overwrite or not os.path.exists(save_path):
            features = self.get_input_data(self.train_file)
            write_pickle(features, save_path)
        else:
            features = load_pickle(save_path)
#它会将特征传递给NERDataset类来构建数据集，然后返回该数据集。
        train_dataset = NERDataset(features)
    #打印出训练数据的长度。
        logger.info('len of train data:{}'.format(len(features)))
        return train_dataset

    def get_dev_data(self):
        logger.info('loading dev data')
        if type(self) == BertProcessor:
            file_name = 'bert_dev.pkl'
        else:
            file_name = 'lebert_dev.pkl'
        save_path = join(self.data_path, file_name)
        if self.overwrite or not os.path.exists(save_path):
            features = self.get_input_data(self.dev_file)
            write_pickle(features, save_path)
        else:
            features = load_pickle(save_path)
#它是通过将features传递给NERDataset类的构造函数创建的对象。  目的是让Bert能够加载
        dev_dataset = NERDataset(features)
        logger.info('len of dev data:{}'.format(len(features)))
        return dev_dataset

    def get_test_data(self):
        logger.info('loading test data')
        if type(self) == BertProcessor:
            file_name = 'bert_test.pkl'
        else:
            file_name = 'lebert_test.pkl'
        save_path = join(self.data_path, file_name)
        if self.overwrite or not os.path.exists(save_path):
            features = self.get_input_data(self.test_file)
            write_pickle(features, save_path)
        else:
            features = load_pickle(save_path)
        test_dataset = NERDataset(features)
        logger.info('len of test data:{}'.format(len(features)))
        return test_dataset

# 吧词语向量的信息加入 相较于bert区别
class LEBertProcessor(Processor):
    def __init__(self, args, tokenizer):
        super(LEBertProcessor, self).__init__(args)
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.data_path = args.data_path #: 数据存储路径。
        self.max_seq_len = args.max_seq_len
        self.max_word_num = args.max_word_num #每个句子中的最大词语数。
        self.overwrite = args.overwrite #是否覆盖之前保存的处理数据。
        self.output_path = args.output_path
        self.tokenizer = tokenizer
        data_files = [self.train_file, self.dev_file, self.test_file]
        self.word_embedding, self.word_vocab, self.label_vocab, self.trie_tree = self.init(
            args.pretrain_embed_path, args.output_path,  args.max_scan_num, data_files, args.label_path, args.overwrite
        )
    #trie_tree：trie树，用于最大匹配分词。
    #args.overwrite: 是否覆盖输出文件
    #data_files: 包含训练、开发、测试数据集的文件路径的字典
#这个函数的作用是从预训练词表中加载所有单词，并将其存储到一个文本文件中
    def init(self, pretrain_embed_path, output_path,  max_scan_num, data_files, label_path, overwrite):
        word_embed_path = join(self.data_path, 'word_embedding.pkl')
        word_vocab_path = join(self.data_path, 'word_vocab.pkl')
        word_vocab_path_ = join(self.data_path, 'word_vocab.txt')
        trie_tree_path = join(self.data_path, 'trie_tree.pkl')
#这段代码是判断是否需要重新生成词向量和词表文件，或者是否已经存在这些文件。
# 如果overwrite为True或者这些文件不存在，就会重新生成这些文件，否则会直接从已有的文件中读取数据。
        if overwrite or not os.path.exists(word_embed_path) or not os.path.exists(word_vocab_path):
            # 加载词向量
            #word_list 则是词典中所有单词的列表
            word_embed_dict, word_list, word_embed_dim = self.load_word_embedding(pretrain_embed_path, max_scan_num)
            # 构建字典树
            trie_tree = self.build_trie_tree(word_list, trie_tree_path)
            # 找到数据集中的所有单词
            corpus_words = self.get_words_from_corpus(data_files, word_vocab_path_, trie_tree)
            # 初始化模型的词向量
            model_word_embedding, word_vocab, embed_dim =self.init_model_word_embedding(corpus_words, word_embed_dict, word_embed_path, word_vocab_path)
        else:
            model_word_embedding = load_pickle(word_embed_path)
            word_vocab = load_pickle(word_vocab_path)
            trie_tree = load_pickle(trie_tree_path)
        # 加载label
        labels = load_lines(label_path)
        label_vocab = Vocabulary(labels, vocab_type='label')
        return model_word_embedding, word_vocab, label_vocab, trie_tree

    @classmethod
    def load_word_embedding(cls, word_embed_path, max_scan_num):
        """
        todo 存在许多单字的，考虑是否去掉
        加载前max_scan_num个词向量, 并且返回词表
        :return:
        """
        logger.info('loading word embedding from pretrain')
        word_embed_dict = dict()
        word_list = list()
        with open(word_embed_path, 'r', encoding='utf8') as f:
            for idx, line in tqdm(enumerate(f)):
                # 只扫描前max_scan_num个词向量
                if idx > max_scan_num:
                    break
                items = line.strip().split()
    #读取第一行时，将前两个元素作为预训练词向量的个数num_embed和每个词向量的维度word_embed_dim。
                if idx == 0:
                    assert len(items) == 2
                    num_embed, word_embed_dim = items
                    num_embed, word_embed_dim = int(num_embed), int(word_embed_dim)
                else:
                    assert len(items) == word_embed_dim + 1
                    word = items[0]
                    embedding = np.empty([1, word_embed_dim])
                    embedding[:] = items[1:]
                    word_embed_dict[word] = embedding
                    word_list.append(word)
        logger.info('word_embed_dim:{}'.format(word_embed_dim))
        logger.info('size of word_embed_dict:{}'.format(len(word_embed_dict)))
        logger.info('size of word_list:{}'.format(len(word_list)))

        return word_embed_dict, word_list, word_embed_dim

    @classmethod
    def build_trie_tree(cls, word_list, save_path):
        """
        # todo 是否不将单字加入字典树中
        构建字典树
        :return:
        """
        logger.info('building trie tree')
        trie_tree = Trie()
        for word in word_list:
            trie_tree.insert(word) # 插入词语 或者词
        write_pickle(trie_tree, save_path)
        return trie_tree

    @classmethod
    def get_words_from_corpus(cls, files, save_file, trie_tree):
        """
        找出文件中所有匹配的单词
        :param files:
        :return:
        """
        logger.info('getting words from corpus')
        all_matched_words = set()
        for file in files:
            with open(file, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for idx in trange(len(lines)):
                    line = lines[idx].strip()
                    data = json.loads(line)
                    text = data['text']
                    matched_words = cls.get_words_from_text(text, trie_tree)
                    _ = [all_matched_words.add(word) for word in matched_words]

        all_matched_words = list(all_matched_words)
        all_matched_words = sorted(all_matched_words)
        write_lines(all_matched_words, save_file)
        return all_matched_words

    @classmethod
    def get_words_from_text(cls, text, trie_tree):
        """
        找出text中所有的单词
        :param text:
        :param trie_tree:
        :return:
        """
        length = len(text)
        matched_words_set = set()  # 存储匹配到的单词
 #从文本的每个位置开始，以当前位置为起点取出一个最大深度为trie_tree的最大深度的子串。此处的最大深度是trie_tree中单词的最大长度。
        for idx in range(length):
            sub_text = text[idx:idx + trie_tree.max_depth]
            words = trie_tree.enumerateMatch(sub_text)
            _ = [matched_words_set.add(word) for word in words]
        matched_words_set = list(matched_words_set)
        matched_words_set = sorted(matched_words_set)
        return matched_words_set

    def init_model_word_embedding(self, corpus_words, word_embed_dict, save_embed_path, save_word_vocab_path):
        logger.info('initializing model word embedding')
  #构建单词到id的映射，并根据预训练好的词向量文件（pretrain_embed_path）将词向量转化为矩阵 model_word_embedding
#如果预训练的词向量中包含该单词，就使用对应的词向量；如果不包含该单词，就用随机初始化的词向量。
        # 同时记录匹配到的单词数和未匹配到的单词数，并保存构建好的词向量矩阵和单词到id的映射。
        # 构建单词和id的映射
        word_vocab = Vocabulary(corpus_words, vocab_type='word')
        # embed_dim = len(word_embed_dict.items()[1].size)
        embed_dim = next(iter(word_embed_dict.values())).size

        scale = np.sqrt(3.0 / embed_dim)
        model_word_embedding = np.empty([word_vocab.size, embed_dim])

        matched = 0
        not_matched = 0

        for idx, word in enumerate(word_vocab.idx2token): # 词向量
            if word in word_embed_dict:
                #即将当前单词的embedding向量填充到这一行的所有列中。
                model_word_embedding[idx, :] = word_embed_dict[word]
                matched += 1
            else:
                model_word_embedding[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
                not_matched += 1

        logger.info('num of match:{}, num of not_match:{}'.format(matched, not_matched))
        write_pickle(model_word_embedding, save_embed_path)
        write_pickle(word_vocab, save_word_vocab_path)

        return model_word_embedding, word_vocab, embed_dim

    def get_char2words(self, text):
        """
        获取每个汉字，对应的单词列表
        :param text:
        :return:
        """
        text_len = len(text)
        char_index2words = [[] for _ in range(text_len)]

        for idx in range(text_len):
            sub_sent = text[idx:idx + self.trie_tree.max_depth]  # speed using max depth
            words = self.trie_tree.enumerateMatch(sub_sent)  # 找到以text[idx]开头的所有单词
            for word in words:
                start_pos = idx
                end_pos = idx + len(word)
                for i in range(start_pos, end_pos):
                    char_index2words[i].append(word)
        # todo 截断
        # for i, words in enumerate(char_index2words):
        #     char_index2words[i] = char_index2words[i][:self.max_word_num]
        return char_index2words

    def get_input_data(self, file):
        lines = load_lines(file)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label_vocab.convert_token_to_id('O')
        pad_label_id = self.label_vocab.convert_token_to_id('[PAD]')

        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            labels = data['label']
            char_index2words = self.get_char2words(text)

            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text) + [sep_token_id]
            label_ids = [o_label_id] + self.label_vocab.convert_tokens_to_ids(labels) + [o_label_id]

            word_ids_list = []
            word_pad_id = self.word_vocab.convert_token_to_id('[PAD]')
            for words in char_index2words:
                words = words[:self.max_word_num]
                word_ids = self.word_vocab.convert_tokens_to_ids(words)
                word_pad_num = self.max_word_num - len(words)
                word_ids = word_ids + [word_pad_id] * word_pad_num
                word_ids_list.append(word_ids)
            # 开头和结尾进行padding
            word_ids_list = [[word_pad_id]*self.max_word_num] + word_ids_list + [[word_pad_id]*self.max_word_num]

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]
                word_ids_list = word_ids_list[: self.max_seq_len]
            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            assert len(input_ids) == len(label_ids) == len(word_ids_list)

            # padding
            padding_length = self.max_seq_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            input_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            label_ids += [pad_label_id] * padding_length
            word_ids_list += [[word_pad_id]*self.max_word_num] * padding_length

            text = ''.join(text)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)
            word_ids = torch.LongTensor(word_ids_list)
            word_mask = (word_ids != word_pad_id).long()

            feature = {
                'text': text, 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids,
                'word_ids': word_ids, 'word_mask': word_mask, 'label_ids': label_ids
            }
            features.append(feature)

        return features

    # def get_train_data(self):
    #     logger.info('loading train data')
    #     save_path = join(self.data_path, 'train.pkl')
    #     if self.overwrite or not os.path.exists(save_path):
    #         features = self.get_input_data(self.train_file)
    #         write_pickle(features, save_path)
    #     else:
    #         features = load_pickle(save_path)
    #     train_dataset = NERDataset(features)
    #     logger.info('len of train data:{}'.format(len(features)))
    #     return train_dataset
    #
    # def get_dev_data(self):
    #     logger.info('loading dev data')
    #     save_path = join(self.data_path, 'dev.pkl')
    #     if self.overwrite or not os.path.exists(save_path):
    #         features = self.get_input_data(self.dev_file)
    #         write_pickle(features, save_path)
    #     else:
    #         features = load_pickle(save_path)
    #     dev_dataset = NERDataset(features)
    #     logger.info('len of dev data:{}'.format(len(features)))
    #     return dev_dataset
    #
    # def get_test_data(self):
    #     logger.info('loading test data')
    #     save_path = join(self.data_path, 'test.pkl')
    #     if self.overwrite or not os.path.exists(save_path):
    #         features = self.get_input_data(self.test_file)
    #         write_pickle(features, save_path)
    #     else:
    #         features = load_pickle(save_path)
    #     test_dataset = NERDataset(features)
    #     logger.info('len of test data:{}'.format(len(features)))
    #     return test_dataset


class BertProcessor(Processor):
    def __init__(self, args, tokenizer):
        super(BertProcessor, self).__init__(args)
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.data_path = args.data_path
        self.max_seq_len = args.max_seq_len
        self.max_word_num = args.max_word_num
        self.overwrite = args.overwrite
        self.output_path = args.output_path
        self.tokenizer = tokenizer

        # 加载label
        labels = load_lines(args.label_path)
        self.label_vocab = Vocabulary(labels, vocab_type='label')

    def get_input_data(self, file):
        lines = load_lines(file)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label_vocab.convert_token_to_id('O')
        pad_label_id = self.label_vocab.convert_token_to_id('[PAD]')

        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            labels = data['label']

            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text) + [sep_token_id]
            label_ids = [o_label_id] + self.label_vocab.convert_tokens_to_ids(labels) + [o_label_id]

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]
            input_mask = [1] * len(input_ids)
#，用来表示输入序列中每个token所属的segment（即第一个句子还是第二个句子）
#在NER任务中，这个列表通常都是一直为0，因为只有一个句子需要进行NER标注。
            token_type_ids = [0] * len(input_ids)
            assert len(input_ids) == len(label_ids)

            # padding
            padding_length = self.max_seq_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            input_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            label_ids += [pad_label_id] * padding_length

            text = ''.join(text)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)

            feature = {
                'text': text, 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids,
                'label_ids': label_ids
            }
            features.append(feature)

        return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_file = '/root/autodl-tmp/project/datasets1/ner_data/resume/train.json'
    args.dev_file = '/root/autodl-tmp/project/datasets1/ner_data/resume/dev.json'
    args.test_file = '/root/autodl-tmp/project/datasets1/ner_data/resume/test.json'
    args.max_seq_len = 150
    args.max_word_num = 5
#一行是词向量总数（8824330），和词向量维度（200）。 从第二行开始，每行是中文词以及它的词向量表示，每一维用空格分隔。
    args.pretrain_embed_path = '/root/autodl-tmp/project/pretrain_model/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'
    args.output_path = '/root/autodl-tmp/project/output/ner'
    args.max_scan_num = 10000
    args.label_path = '/root/autodl-tmp/project/datasets1/ner_data/resume/labels.txt'
    args.data_path = '/root/autodl-tmp/project/datasets1/ner_data/resume'
    args.overwrite = True
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/project/pretrain_model/bert-base-chinese')
    processor = LEBertProcessor(args, tokenizer)
    train_set = processor.get_dev_data()
    # processor = BertProcessor(args, tokenizer)
    # train_set = processor.get_dev_data()
    # word_embedding_path=args.pretrain_embed_path
    # with open(word_embedding_path, 'r', encoding='utf8') as f:
    #     for idx, line in tqdm(enumerate(f)):
    #         # 只扫描前max_scan_num个词向量
    #         if idx > 1000000:
    #             break
    #         items = line.strip().split()
#CnerProcessor类是用于处理中文命名实体识别数据集的处理器。
class CnerProcessor():
    """Processor for the chinese ner data set."""
    def __init__(self, train_path, dev_path, test_path, tokenizer, max_len, segment_a_id=0):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.segment_a_id = segment_a_id
        self.label2id = {label: i for i, label in enumerate(self.get_labels())}  # 每种标签对应的id
        self.id2label = {i: label for i, label in enumerate(self.get_labels())}  # 每种标签对应的id

    def get_train_examples(self):
        examples = self.create_examples(self.read_text(self.train_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of train data:{}'.format(len(features)))
        return features

    def get_dev_examples(self):
        examples = self.create_examples(self.read_text(self.dev_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of dev data:{}'.format(len(features)))
        return features

    def get_test_examples(self):
        examples = self.create_examples(self.read_text(self.test_path))
        features = self.convert_examples_to_inputs(examples)
        logger.info('len of test data:{}'.format(len(features)))
        return features

    def convert_examples_to_inputs(self, examples):
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        o_label_id = self.label2id['O']
        features = []
        for words, labels in examples:
            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(words) + [sep_token_id]
            label_ids = [o_label_id] + [self.label2id[x] for x in labels] + [o_label_id]
            if len(input_ids) > self.max_len:
                input_ids = input_ids[: self.max_len]
                label_ids = label_ids[: self.max_len]
            input_mask = [1] * len(input_ids)
            token_type_ids = [self.segment_a_id] * len(input_ids)
            assert len(input_ids) == len(label_ids)

            # 对输入进行padding
            padding_length = self.max_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            input_mask += [pad_token_id] * padding_length
            token_type_ids += [pad_token_id] * padding_length
            label_ids += [pad_token_id] * padding_length
            text = ''.join(words)
            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            token_type_ids = torch.LongTensor(token_type_ids)
            feature = {'text': text, 'input_ids': input_ids, 'label_ids': label_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
            features.append(feature)
        return features

    def read_text(self, file):
        """
        读取文件，将每条记录读取为words:['我','在','天','津']，labels:['O','O','B-ORG','I-ORG']
        :param file:
        :return:
        """
        lines = []
        with open(file, 'r', encoding='utf8') as f:
            words = []
            labels = []
            for line in f:
                if line == "" or line == "\n":
                    # 读取完一条记录
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    def get_labels(self):
        """See base class."""
        # return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
        #         'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
        #         'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]
        return ['B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'S-NAME', 'S-ORG', 'S-RACE', 'O']

    def create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            words = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append((words, labels))
        return examples
