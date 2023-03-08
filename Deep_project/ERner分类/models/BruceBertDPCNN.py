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
        self.model_name = "BruceBertDPCNN"
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
        # 卷积核数量
        self.num_filters = 250
        # droptout
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size))

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(3,1), stride=2)

        self.padd1 = nn.ZeroPad2d((0,0,1,1))
        self.padd2 = nn.ZeroPad2d((0,0,0,1))
        self.relu = nn.ReLU()

        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask = mask, output_all_encoded_layers = False)
        out = encoder_out.unsqueeze(1) #[batch_size, 1, seq_len, embed]
        out = self.conv_region(out) #[batch_size, 250, seq_len-3+1, 1]

        out = self.padd1(out) #[batch_size, 250, seq_len,1]
        out = self.relu(out)
        out = self.conv(out) #[batch_size, 250, seq_len-3+1,1]
        out = self.padd1(out)  # [batch_size, 250, seq_len,1]
        out = self.relu(out)
        out = self.conv(out)  # [batch_size, 250, seq_len-3+1,1]
        while out.size()[2] > 2:
            out = self._block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out

    def _block(self, x):
        x = self.padd2(x)
        px = self.max_pool(x)
        x = self.padd1(px)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padd1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        return x


























