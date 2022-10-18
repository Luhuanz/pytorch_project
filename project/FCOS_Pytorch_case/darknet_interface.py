# -*- coding: utf-8 -*-
# CreateBy: liaorongfan
# CreateAT: 2020/9/13
# =====================
import torch
import torch.nn as nn
from model.fcos import FCOSDetector

batch_imgs = torch.rand((3, 3, 320, 480)).cuda()
batch_boxes = torch.ones((3, 4, 4)).type(torch.LongTensor).cuda()
batch_classes = torch.ones((3, 4)).type(torch.LongTensor).cuda()
# torch.Size([3, 3, 1056, 1056]) torch.Size([3, 4, 4]) torch.Size([3, 4])
model = FCOSDetector(mode="training").cuda()
losses = model([batch_imgs, batch_boxes, batch_classes])
loss = losses[-1]

print(loss)
