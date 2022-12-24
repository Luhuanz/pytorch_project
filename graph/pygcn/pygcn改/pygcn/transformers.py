import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim,dropout=dropout)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = x.unsqueeze(0) #torch.Size([1, 2708, 1433])

        adj=adj.unsqueeze(0) #torch.Size([1, 2708, 2708])

        x = self.embedding(x)

        x = self.pos_encoding(x) #torch.Size([1, 2708, 16])

        for layer in self.layers:
            x = layer(x, adj)
        x = self.out(x)

        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.self_attention = SelfAttention(input_dim, hidden_dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, adj):
        x = self.self_attention(x, adj)

        x = self.ffn(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, adj):
        #torch.Size([1, 2708, 16])
        # 计算query, key, value
        adj = adj.to_dense()
        query = self.query(x) #torch.Size([1, 2708, 16])

        key = self.key(x)
        value = self.value(x)

        # 计算注意力权值
        scores = torch.matmul(query, key.transpose(-1, -2))

        scores = scores / (self.hidden_dim ** 0.5)
        scores = torch.where(adj > 0, scores, -1e9 * torch.ones_like(scores))

        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=self.dropout, training=self.training)

        # 计算输出
        output = torch.einsum('ijk,ikl->ijl', scores, value)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
if __name__ == '__main__':
    import torch

    # 创建两个张量
    x = torch.randn(1, 2708, 16)
    y = torch.randn(1, 2708, 16)

    # 计算点积
    z = torch.einsum('ijk,ikl->ijl', x, y)