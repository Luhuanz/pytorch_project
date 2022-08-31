import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt #数据可视化
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

learning_rate = 0.01
batch_size = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 将数据集合下载到指定目录下,这里的transform表示，数据加载时所需要做的预处理操作
# 加载训练集合
train_dataset = torchvision.datasets.MNIST(
    root='.data',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),  # 转化为tensor且数据标准化注意在FC中我们未做数据增强
    download=True)
# 加载测试集合
test_dataset = torchvision.datasets.MNIST(
    root='.data',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,  # 一个批次的大小为128张
    shuffle=True  # 随机打乱

)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


class LeNet(nn.Module):
    # 定义构造方法函数，用来实例化
    def __init__(self):
        super(LeNet, self).__init__()  # 5层， 2个卷积层+3个fc全连接层
        # 1*1*28*28
        # 第一个卷积层：输入通道数=1，输出通道数=6，卷积核大小=5*5，默认步长=1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # 6 * 24 * 24
        # 第二个卷积层：输入通道数=6，输出通道数=16，卷积核大小=5*5，默认步长=1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # 16 * 8 * 8

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)  # 16 * 4 * 4
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        # 第一个全连接层：输入特征数=256，输出特征数=120
        # 也可以理解成：将256个节点连接到120个节点上
        # 第三个全连接层：输入特征数=84，输出特征数=10（这10个维度我们作为0-9的标识来确定识别出的是那个数字。）
        # 也可以理解成：将84个节点连接到10个节点上

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x,dim=1) # 这个问题前面FC建模已经说过了 参考官方LeNet
        return x

def train(epoch):
    run_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):#循环次数batch_idx的最大循环值 +1 = (MNIST数据集样本总数60000/ BATCH_SIZE )
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()#反向计算梯度
        optimizer.step()#优化参数
        run_loss += loss.item()
        if batch_idx % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, run_loss /200))
            run_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():#此时已经不需要计算梯度，也不会进行反向传播
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) #将数据转移到cuda上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)# 将输出结果概率最大的作为预测值，找到概率最大的下标,输出最大值的索引位置
            total += labels.size(0)
            correct += (predicted == labels).sum().item()#正确率累加
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total

if __name__ == '__main__':
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_list = []
    acc_list = []
    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)
    plt.plot(epoch_list, acc_list)
    plt.title("The Mnist on LeNet-5")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    acc = np.array(acc_list).mean()
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    torch.save(model.state_dict(), 'LeNet.pth')



























torch.save(model.state_dict(), 'mlp.pth')