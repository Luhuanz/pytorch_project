#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 微信公众号 AI壹号堂 欢迎关注
# Author 杨博

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        # 模型名称
        self.model_name = "BruceBertRCNN"
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]

        # 模型保存路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 128
        # 序列长度
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # 预训练位置
        self.bert_path = './bert_pretrain'
        # bert的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained((self.bert_path))
        # Bert的隐藏层数量
        self.hidden_size = 768
        # RNN隐层层数量
        self.rnn_hidden = 256
        # rnn数量
        self.num_layers = 2
        # droptout
        self.dropout = 0.5

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)


    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask = mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = F.relu(out)
        out = out.permute(0,2,1)
        out = self.maxpool(out)
        out = out.squeeze()
        out = self.fc(out)
        return out






















