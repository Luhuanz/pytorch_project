import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MNN(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(MNN, self).__init__()
        self.encode0 = nn.Linear(node_size, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size)
        self.droput = droput
        self.alpha = alpha

    def forward(self, adj_batch, adj_mat, b_mat):
        t0 = F.leaky_relu(self.encode0(adj_batch))#torch.Size([100, 1000])
        t0 = F.leaky_relu(self.encode1(t0)) #torch.Size([100, 128])
        embedding = t0 #torch.Size([100, 128])

        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0)) #torch.Size([100, 2708])
        #
        # print(t0.shape)
        # exit()
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)#torch.Size([100, 1])
        print(embedding_norm.shape)
        exit()
        L_1st = torch.sum(adj_mat * (embedding_norm -
                                     2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                                     + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))
        return L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd

    def savector(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0



