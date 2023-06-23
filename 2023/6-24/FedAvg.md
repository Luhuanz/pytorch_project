#  FedAvg

FedAvg 算法将多个使用 SGD 的深度学习模型整合成一个全局模型。

与单机机器学习类似，联邦学习的目标也是经验风险最小化，即

> $$
> \min _{\boldsymbol{x} \in \mathbb{R}^d}\left[F(\boldsymbol{x})=\frac{1}{n} \sum_{i=1}^n f\left(\boldsymbol{x} ; s_i\right)\right]
> $$
>
> 其中，$n$是样本容量，$s_i$表示第$i$个样本个体，$f\left(\boldsymbol{x}, s_i\right)$表示模型在$s_i$上的损失函数。假设有$K$个局部模型，$\mathcal{P}_k$表示第$k$个模型拥有的样本个体的序号集合。令$n_k=\left|\mathcal{P}_k\right|$，我们可以把目标函数重写为
> $$
> \begin{aligned}
> F(\boldsymbol{x}) & =\sum_{k=1}^K \frac{n_k}{n} F_k(\boldsymbol{x}), \\
> F_k(\boldsymbol{x}) & =\frac{1}{n_k} \sum_{i \in \mathcal{P}_k} f\left(\boldsymbol{x} ; s_i\right) .
> \end{aligned}
> $$
> 值得注意的是，由于每个终端设备的数据不能代表全局数据，我们不能认为$\mathbb{E}_{\mathcal{P}_k}\left[F_k(\boldsymbol{x})\right]$
>
> 与$F(\boldsymbol{x})$ 相同，也就是说，任何一个局部模型不能作为全局模型。
>
> 我们将局部模型的一次参数更新称为一次迭代。用$b$表示一个 batch，那么第$k$个局部模型迭代公式为
> $$
> \boldsymbol{x}_k \leftarrow \boldsymbol{x}_k-\frac{\eta}{|b|} \sum_{i \in b} \nabla f\left(\boldsymbol{x}_k ; s_i\right)
> $$
> FedAvg 算法的思想很直观，将训练过程分为多个回合，每个回合中选择$CK$(0≤C≤1)个局部模型对数据进行学习。



