import torch
from collections import Counter
from utils.get_entity import get_entities

class SeqEntityScore(object):
#BIOS" 标记方案是一种常见的实体标注方案，其中 "B" 表示实体的开始， "I" 表示实体的中间部分， "O" 表示非实体部分，而 "S" 则表示只有一个单独的实体。
    def __init__(self, id2label,markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
#计算了一个实体标注模型的精确度（precision）、召回率（recall）和 F1 分数（F1-score）。
# 具体地，这里的参数 origin 表示数据集中的实际实体数，found 表示模型预测出的实体数，right 表示模型预测正确的实体数。
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        #统计数据集中每种类型实体的数量 每个实体的类型（即 B-xxx、I-xxx、O 等）
        origin_counter = Counter([x[0] for x in self.origins])
        #统计模型预测出的每种类型实体的数量。
        found_counter = Counter([x[0] for x in self.founds])
        #统计模型预测正确的每种类型实体的数量。
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
#将计算得到的每个类型实体的召回率、精确度和 F1 分数保存到字典 class_info 中。 四舍五入处理
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label,self.markup)
            pre_entities = get_entities(pre_path, self.id2label,self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


