# torch.einsum()

```python
import torch

# 创建两个 2x3 的张量
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# 计算两个张量的点积
c = torch.einsum('ij,ij->', a, b)

print(c)

```

`ij,ij->` 表示将a和b的相同位置上的元素相乘，然后对所有结果求和，得到一个标量值。

"ij" 表示矩阵的行列索引，其中 "i" 为行索引，"j" 为列索引。例如，对于一个 $3 \times 3$ 的矩阵 $A$，$A_{21}$ 表示第二行第一列的元素。在 `torch.einsum()` 中，可以通过 "ij" 来表示矩阵的所有元素。

Einsum 函数

ein 就是爱因斯坦的 ein，sum 就是求和。einsum 就是爱因斯坦求和约定，其实作用就是把求和符号省略，就这么简单。举个例子：

我们现在有一个矩阵

我们想对 A 的 “行” 进行求和得到矩阵 B (向量 B)，用公式表示，则为：
$$
B_i=\sum_j A_{i j}=B_2=\left(\begin{array}{l}
3 \\
7
\end{array}\right)
$$
对于这个求和符号，爱因斯坦说看着有点多余，要不就省略了吧，然后式子就变成了:

用 einsum 表示呢，则为: `torch.einsum("ij->i", A)`。`->` 符号就相当于等号，`->` 左边的 `ij` 就相当于$A_{ij}$`->` 右边的 `i` 就相当于$B_i$。`einsum` 接收的第一个参数为 einsum 表达式，后面的参数为等号右边的矩阵。

```python
A = torch.Tensor(range(2*3*4)).view(2, 3, 4)
C = torch.einsum("ijk->jk", A)

```

则，该式子的数学表达式为：
$$
C_{j k}=A_{i j k}
$$
![image-20230413080030934](torch.einsum().assets/image-20230413080030934.png)

![image-20230413080045286](torch.einsum().assets/image-20230413080045286.png)

# 看懂一个 einsum 式子

