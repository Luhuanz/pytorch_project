import torch
import torch.nn as nn
import torch.nn.functional as F

# 继承自nn.Module
class Line(nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")
        # 设置embedding的维度
        self.embed_dim = embed_dim
        self.order = order
        # nodes数*embedding维度
        self.nodes_embeddings = nn.Embedding(size, embed_dim)

        # 初始化模型参数
        # 只有1st order时每个node只需要一个embedding
        # 当有2nd order时每个node还需要一个context embedding（邻居），共计两个
        if order == 2:
            self.contextnodes_embeddings = nn.Embedding(size, embed_dim)
            # uniform的Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim

        # uniform的Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i).to(device)

        # 这里是1阶2阶相似度计算的区别，2阶是用上下文contextnodes_embeddings
        # 1阶用的是nodes_embeddings
        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

        else:
            v_j = self.nodes_embeddings(v_j).to(device)
            negativenodes = -self.nodes_embeddings(negsamples).to(device)

        # 公式（7）中的第一项（正样本计算），第一步是点乘，然后是按行求和
        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))
        # 公式（7）中的第二项（负样本计算）
        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch, dim=2)),dim=1)
        loss = positivebatch + negativebatch
        return -torch.mean(loss)
