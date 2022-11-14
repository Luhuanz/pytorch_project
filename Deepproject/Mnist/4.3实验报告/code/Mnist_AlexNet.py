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
batch_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 将数据集合下载到指定目录下,这里的transform表示，数据加载时所需要做的预处理操作
# 加载训练集合
train_dataset = torchvision.datasets.MNIST(
    root='.data',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224), 
        transforms.Normalize((0.1307,), (0.3081,))
    ]),  # 转化为tensor且数据标准化注意在FC中我们未做数据增强
    download=True)
# 加载测试集合
test_dataset = torchvision.datasets.MNIST(
    root='.data',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224), 
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


class AlexNet(nn.Module):
       # 定义构造方法函数，用来实例化
    def __init__(self):
        super(AlexNet, self).__init__()  #  
        self.conv1=nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1)
        self.conv2=nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        #接下来连续3分卷积层和较小的卷积窗口
        self.conv3=nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.conv5=nn.Conv2d(384,256,kernel_size=3,padding=1)
        # FC层
        self.fc1=nn.Linear(6400,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,10)
        # AlexNet本来是做1000分类的，我们采用迁移学习的思想该为10fc3
    def forward(self, x):                   # 输入shape: torch.Size([1, 1, 224, 224])
        x=F.max_pool2d(F.relu(self.conv1(x)),kernel_size=3,stride=2)
        x=F.max_pool2d(F.relu(self.conv2(x)),kernel_size=3,stride=2)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.max_pool2d(F.relu(self.conv5(x)),kernel_size=3,stride=2)
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,p=0.5)
        x=F.relu(self.fc2(x))
        x=F.dropout(x,p=0.5)
        x=self.fc3(x)
        return x

def train(epoch):
    run_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0): 
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
    with torch.no_grad(): 
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total

if __name__ == '__main__':
    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum = 0.9) 
    epoch_list = []
    acc_list = []
    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)
    plt.plot(epoch_list, acc_list)
    plt.title("The Mnist on AlexNet")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    acc = np.array(acc_list).mean()
    print(f'Accuracy of the network on the 10000 test images: {100*acc} %')
    torch.save(model.state_dict(), 'AlexNet.pth')


























 