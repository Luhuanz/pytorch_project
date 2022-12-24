import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch
from pygcn.transformers import GraphTransformer
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear=nn.Linear(nclass,nclass)
        self.trans=GraphTransformer(nfeat,nhid,nclass,4,dropout)
    def forward(self, x, adj):
        x__=self.trans(x,adj) #torch.Size([1, 2708, 7])
        x__=x__.squeeze()
        x = F.relu(self.gc1(x, adj)) #torch.Size([2708, 16])

        x = F.dropout(x, self.dropout, training=self.training) #torch.Size([2708, 16])
        x = self.gc2(x, adj) #torch.Size([2708, 7])
        U, S, V = torch.svd(x)
        U_ = torch.nn.functional.normalize(U, dim=1)
        U_=torch.abs(U_)
        x_ = U_ * x
        x_= self.linear(x_)
        x_=F.relu(x)
        x =  x +x__+0.1*x_

        return F.log_softmax(x, dim=1)
