"""
Take Performer as T2T Transformer
"""
import math
import torch
import torch.nn as nn

class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)  # 147  3*64=192
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)


       # ---------------------
        # print((x * x).shape) torch.Size([2, 3136, 64]) k或者q
        # exit()
        # print((x * x).sum(dim=-1, keepdim=True).shape) torch.Size([2, 3136, 1])
        # exit()
        # print(((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m).shape)torch.Size([2, 3136, 32])
        # exit()
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2 #torch.Size([2, 3136, 32])
        # print(xd.shape)
        # exit()
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w) #torch.Size([2, 3136, 32])
        # print((torch.exp(wtx - xd) / math.sqrt(self.m)).shape) #torch.Size([2, 3136, 32])
        #
        # exit()
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        # print(x.shape) torch.Size([2, 3136, 147])
        # exit()
        # print(self.kqv(x).shape) torch.Size([2, 3136, 192])
        # exit()
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1) # 第一个按照最后一维 分64
        # print(k.shape) torch.Size([2, 3136, 64])
        # exit()
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        # print(kp.shape) #torch.Size([2, 3136, 32])
        # exit()
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        # print(D.shape) torch.Size([2, 3136, 1])
        # exit()
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        # print(kptv.shape) #torch.Size([2, 64, 32])
        # exit()
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # print(y.shape) #torch.Size([2, 3136, 64])
        # exit()
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        # print(y.shape)torch.Size([2, 3136, 64])
        # exit()
        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # print(x.shape)
        # exit()
        return x

