import torch
import torch.nn as nn
import numpy as np
import math
from torchvision.utils import save_image
from torch.utils.data import  DataLoader
from torchvision import datasets
import  argparse
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import os
os.makedirs('image',exist_ok=True)
device= 'cuda' if torch.cuda.is_available() else 'cpu'
parser=argparse.ArgumentParser()
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt=parser.parse_args()
print(opt)
image_shape=(opt.channels,opt.img_size,opt.img_size)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
    def blocks(in_peat,out_peat,normalize=True,T_right=False):
        layers= [nn.Linear(in_peat,out_peat,bias=True)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_peat,eps=0.8))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        return layers

    self.model=nn.Sequential(
        *blocks(in_peat=opt.latent_dim,out_peat=128,normalize=False),
        *blocks(128,256),
        *blocks(256,512),
        *blocks(512,int(np.prod(opt.img_size))),
        nn.Tanh()
    )
    if T_right:
        self._inweignts()
    def _inweignts(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    nn.init.constant(m.bias)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.normal_(m.weight)

    def forward(self,x):
        x=self.model(x)
        x=x.view(x.size(0),*image_shape) # batch, c,h,w
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self,x):

        x=x.view(x.size(0),-1)
        x = self.model(x)
        return x
# 数据准备
transforms=transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize([0.5],[0.5]),
             transforms.Resize(opt.img_size)
            ]
                              )
train_data=datasets.MNIST(root='mnist',train=True,download=True,transform=transforms,batch_size=opt.batch_size,shuffle=True)
# test_data=datasets.MNIST(root='mnist',train=False,download=True,transform=transforms.ToTensor(),drop_last=True) 不需要验证集 因为不是分类任务而是生成
Train_dataloader=DataLoader(train_data)
criterion=nn.BCELoss().to(device)
#model
generator=Generator().to(device)
discriminator=Discriminator().to(device)
optimize_G=optim.Adam(generator.parameters(),lr=opt.lr,betas=[opt.b1,opt.b2])
optimize_D=optim.Adam(discriminator.parameters(),lr=opt.lr,betas=[opt.b1,opt.b2])
for i in range(opt.epochs+1):
    avg_total=0
    for j,(imgs,_) in enumerate(Train_dataloader):
        #  print(imgs.shape) #torch.Size([64, 1, 28, 28])
        #  exit()
        #  vaild=torch.FloatTensor(1.0,requires_grad=False).expand(imgs.size(0),1).to(device)
        #  fake=torch.FloatTensor(0.0,requires_grad=False).expand(imgs.size(0),1).to(device)
        #  real_imgs=imgs.type(torch.Tensor)
        #  optimize_G.zero_grad()
        #  z=torch.Tensor(np.random.normal(0,1,(imgs.shape[0],opt.latent.dim)))
        #  gen_imgs=generator(z)
        #  loss=criterion(discriminator(gen_imgs),vaild)
        #  loss.backward()
        #  optimize_G.step()
        # # 判别器
        #  optimize_D.zero_grad()
        #  real_loss=criterion(discriminator(real_imgs),valid)
        #  fake_loss=criterion(discriminator(gen_imgs.detach()),fake)
        real_imgs=imgs.type(torch.Tensor)
        vaild=torch.FloatTensor(1.0,requires_grad=False).expand(imgs.size(0),1)
        fake=torch.FloatTensor(0.0,requires_grad=False).expand(imgs.size(0),1)
        # 训练生成器
        random_z=torch.Tensor(np.random.normal(0,1,(imgs.size(0),opt.latent.dim)))
        gen_imgs=generator(random_z)
        optimize_G.zero_grad()
        loss=criterion(discriminator(gen_imgs),vaild)
        loss.backward()
        optimize_G.step()

        # 训练判别器
        optimize_D.zero_grad()
        fake_loss=criterion(discriminator(gen_imgs),fake)
        real_loss=criterion(discriminator(real_imgs),vaild)


