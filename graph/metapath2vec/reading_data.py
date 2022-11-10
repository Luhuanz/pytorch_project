import numpy as np
import torch
from torch.utils.data import Dataset
from download import AminerDataset

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, dataset, min_count, care_type):

        # 初始化变量
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = dataset.fn

        # 执行函数，给初始化变量赋值
        # 读metapath文件
        self.read_words(min_count)
        # 产生负采样节点列表
        self.initTableNegatives()
        # word2vec中的subsampling
        self.initTableDiscards()

    def read_words(self, min_count):
        # 统计词频，将出现次数少于min_count(如5)的单词/节点过滤掉
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="ISO-8859-1"):
            line = line.split()
            if len(line) > 1:
                # 统计有多少个句子/这里一个句子是一次random walk(metapath sequences)
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        # 统计词数/节点数，统计重复的
                        self.token_count += 1
                        # 统计词频
                        word_frequency[word] = word_frequency.get(word, 0) + 1
                        # 输出读图过程
                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            # 统计词频，将出现次数少于min_count(如5)的单词/节点过滤掉
            if c < min_count:
                continue
            # word/node name -> id
            self.word2id[w] = wid
            # id -> word/node name
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        # 注意留下的词全是词频 >= min_count的
        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.

        # word2vec中的subsampling
        # 最高频的词汇，比如in, the, a这些词。这样的词汇通常比其它罕见词提供了更少的信息量。
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        # http://d0evi1.com/word2vec-subsampling/
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        # 举例: self.negatives = [1 1 1 2 2 2 2 3 3 3 4 4 4]
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        # ratio是每个词/节点负采样的概率
        ratio = pow_frequency / words_pow
        # 每个词/节点根据ratio看应该分配多少个
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        # 产生对应的negatives列表
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        # 打乱顺序
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            # negpos初始为0
            # 取出size个负样本
            response = self.negatives[self.negpos:self.negpos + size]
            # 挪动negpos的位置，方便下一次取negative samples
            self.negpos = (self.negpos + size) % len(self.negatives)
            # 处理negatives列表已经遍历到结尾的情况
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

# -----------------------------------------------------------------------------------------------------------------

class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        # 窗口大小; parser读入
        self.window_size = window_size
        # sampler.py的输出文件: output_path.txt，即metapath sequences
        self.input_file = open(data.inputFileName, encoding="ISO-8859-1")

    # https://zhuanlan.zhihu.com/p/30385675
    def __len__(self):
        # return the number of walks
        # 统计有多少个句子/这里一个句子是一次random walk(metapath sequences)
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            # https://www.runoob.com/python/file-seek.html
            # 当文件读完的时候，重新回到文件的开头开始读取
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    # "w in self.data.word2id": 词频>=min_count
                    # "discards": 满足word2vec中的subsampling
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
                    # 打包成一组带训练的数据:(u, v, [n1, n2, n3, n4, n5])
                    # 其中n1, n2, n3, n4, n5是一个个negative sample
                    pair_catch = []
                    # 遍历每一个节点u
                    for i, u in enumerate(word_ids):
                        # 对每一个节点u的上下文节点v组成正样本pair (u, v)
                        # v的范围长度由window_size决定
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append((u, v, self.data.getNegatives(v, 5)))
                    return pair_catch

    # 转化成Tensor
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
