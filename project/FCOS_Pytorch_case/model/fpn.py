'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import torch.nn as nn
import torch.nn.functional as F
import math


class FPN_(nn.Module):
    """only for resnet50,101,152"""

    def __init__(self, features=256, use_p5=True):
        super(FPN, self).__init__()
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)  # 对FPN结构使用凯明初始化

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                             mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]


class FPN(nn.Module):
    """only for resnet50,101,152"""

    def __init__(self, features=256, use_p5=True, backbone="resnet50"):
        super(FPN, self).__init__()
        if backbone == "resnet50":
            print("resnet50 backbone")
            self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)  # 不改变特征图的尺寸
            self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        elif backbone == "darknet19":
            print("darnet19 backbone")
            self.prj_5 = nn.Conv2d(1024, features, kernel_size=1)  # 不改变特征图的尺寸
            self.prj_4 = nn.Conv2d(512, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(256, features, kernel_size=1)
        else:
            raise ValueError("arg 'backbone' only support 'resnet50' or 'darknet19'")

        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)  # 不改变特征图的尺寸
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)  # 将特征图尺寸缩小一半
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)  # 将特征图尺寸缩小一半
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)  # 将特征图尺寸缩小一半
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)  # 对FPN结构使用凯明初始化

    def upsamplelike(self, inputs):  # 将src的尺寸大小，上采样到 target的尺寸
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):  # 判断变量module是不是nn.Conv2d类
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        # print(C3.shape, C4.shape, C5.shape)
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])  # 先将P5上采样到C4大小，再用元素相加的方式进行融合
        P3 = P3 + self.upsamplelike([P4, C3])  # 先将P4上采样到C3大小，再用元素相加的方式进行融合

        P3 = self.conv_3(P3)  # 融合后再卷积的目的：用卷积操作平滑一下特征图的数值
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]   # 返回融合后的特征图
