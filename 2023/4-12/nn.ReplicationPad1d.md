# nn.ReplicationPad1d

`nn.ReplicationPad1d((0, padding))` 的计算公式如下：

给定一维输入张量 $x$ 和一维 padding 的大小为 $padding$，则对于输入 $x$ 的每个维度 $d$，该操作会将 $x$ 的第 $d$ 维复制扩展成 $2 \times padding + x.\text{size}(d)$ 个元素，然后在两端填充 $padding$ 个元素。

例如，如果 $x$ 的形状为 $(N, C, L)$，其中 $L$ 表示输入序列的长度，则应用 `nn.ReplicationPad1d((0, padding))` 后，输出的形状为 $(N, C, L + 2 \times padding)$，其中 $L + 2 \times padding$ 表示扩展后的序列长度。

具体来说，对于第 $d$ 维的一个大小为 $L_d$ 的输入 $x$，假设 padding 的大小为 $padding$，则扩展后的序列为：

$x_i^{\prime}= \begin{cases}x_{\max (i-2 \times \text { padding }, 0)}, & i<\text { padding } \\ x_{\min \left(i-2 \times \text { padding }, L_d-1\right)}, & \text { padding } \leq i<L_d+\text { padding } \\ x_{\max \left(i-\left(L_d+2 \times \text { padding-1),0) }\right.\right.}, & L_d+\text { padding } \leq i\end{cases}$

其中 $x^\prime$ 表示扩展后的序列，$x$ 表示输入序列，$i$ 表示扩展后序列的下标。