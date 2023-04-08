import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        ## 定义掩码形状
        mask_shape = [B, 1, L, L]
        ## 禁用梯度计算，以减少计算量
        with torch.no_grad():
            ## 生成上三角矩阵，并将其转换为布尔型张量
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
## 返回掩码
    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        ## 生成全为 True 的上三角矩阵
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        ## 扩展矩阵形状
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        ## 生成一个指示矩阵，表示哪些位置需要保留，哪些位置需要掩盖
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        ## 重新调整形状
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
if __name__ == '__main__':


    # 实例化 TriangularCausalMask 类
    B = 2
    L = 4 #sen_length
    mask = TriangularCausalMask(B, L)

    # 输出掩码的形状
    print(mask.mask.shape)  # torch.Size([2, 1, 4, 4])

    # 实例化 ProbMask 类
    H = 2 #H 是指注意力头的数量，
    index = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]) #batch_szie  2 sen_length/time_step 4
    #index是一个形状为 (B, H, L) 的张量，其中 B 是 batch size，H 是注意力头数，L 是序列长度，用来表示当前注意力头的查询向量在序列中的位置。
    scores = torch.randn(B, H, L, L)
    prob_mask = ProbMask(B, H, L, index, scores)

    # 输出掩码的形状
    print(prob_mask.mask.shape)  # torch.Size([2, 2, 4, 4])