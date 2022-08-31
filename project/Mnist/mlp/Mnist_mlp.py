import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

batch_size=128
input_size = 784
num_out = 10
num_epochs=6# 我们跑6个周期
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.01 #设置学习率为0.01
# 将数据集合下载到指定目录下,这里的transform表示，数据加载时所需要做的预处理操作
# 加载训练集合
train_dataset=torchvision.datasets.MNIST(
                                        root='.data',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),#转化为tensor
                                        download=True)
# 加载测试集合
test_dataset=torchvision.datasets.MNIST(
                                         root='.data',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_loader=torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, #一个批次的大小为128张
        shuffle=True  #随机打乱
        )

test_loader=torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
        )

class MLPNet(nn.Module):
    # 输入数据的维度，中间层的节点数，输出数据的维度
    def __init__(self, input_size, hidden_size, num_out):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.h1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.out = nn.Linear(hidden_size, num_out)
        self.softmax=nn.Softmax(dim=1)# dim表示一维输出

    def forward(self, x):
        h1 = self.h1(x)
        a1 = self.relu1(h1)
        out = self.out(a1)
      #  a_out=self.softmax(out) 这个是没必要的因为我们要把out进行后项传播
        return out
# 建立了一个中间层为 300 的节点三层神经网络，且将模型转为当前环境支持的类型（CPU 或 GPU）
model = MLPNet(input_size, 300, num_out).to(device)
criterion = nn.CrossEntropyLoss()  #交叉熵损失函数，一般分类问题都用 one-hot + Cross-entropy 回归问题 用MSE
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)# 定义 Adam 优化器用于梯度下降 当然也可以用SGD
def train(epoch):
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (x_images, y_labels) in enumerate(train_loader):
            # 因为全连接会把一行数据当做一条数据，因此我们需要将一张图片转换到一行上
            # 原始数据集的大小: [ 100, 1，28, 28]
            # resize 后的向量大小: [-1, 784]
            images = x_images.reshape(-1, 28 * 28).to(device)
            labels = y_labels.to(device)
            # 正向传播以及损失
            y_pre = model(images)  # 前向传播
            loss = criterion(y_pre, labels)  # 计算损失函数
            # 反向传播
            # 梯度清空，反向传播，权重更新
            optimizer.zero_grad()  # 梯度归零 因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。
            loss.backward()
            optimizer.step()  # 执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。

            if (i + 1) % 64 == 0:
                print(f'epoch [{epoch + 1}/{num_epochs}], step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    print("模型训练完成")
def test():
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            # 和训练代码一致
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # 进行模型训练的准确度
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')
if __name__ == '__main__':
    print('Using Device:', device)
    train(6)
    test()
    torch.save(model.state_dict(), 'mlp.pth')