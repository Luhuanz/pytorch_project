import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        # 节点个数
        self.emb_size = emb_size
        # 维度
        self.emb_dimension = emb_dimension
        # 参数矩阵(node)
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        # 参数矩阵(contex node)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        # 初始化
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        # 取出对应的batch embedding
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        # score = [batch, 1]
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        # emb_neg_v = [batch, 5, dim]
        # emb_u = [batch, dim]
        # emb_u.unsqueeze(2) = [batch, dim, 1]
        # torch.bmm(emb_neg_v, emb_u.unsqueeze(2)) = [batch, 5, 1]
        # torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze() = [batch, 5]

        # [batch, 5]
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)

        # [batch, 1]
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # 求平均
        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            # 第一行：节点个数，维度
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            # 节点及对应的维度
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))