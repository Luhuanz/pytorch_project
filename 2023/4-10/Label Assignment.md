# **Label Assignment**

**Label Assignment 定义**

> Label/Target Assignment 顾名思义，提供**学习目标的表示**，“传授” 网络正确与错误的概念，使网络对相应的输入输出正确期望的目标 (如类别、目标位置等等)。Label Assignment 直接决定了网络是否能正确执行任务。如在目标检测任务中，以下图为例，我们希望网络能输出人的目标检测框，此时假设网络输出三个检测框，分别是绿色、蓝色和红色。显然我们需要告诉网络绿色是最合适的框，而不是包裹不完全的蓝色框或者位置偏移的红色框。“告诉” 这一动作其实就是 Label Assignment 的任务。
>
> ![8648c70a0323b282e95882dc79db6292.png](https://img-blog.csdnimg.cn/img_convert/8648c70a0323b282e95882dc79db6292.png)
>
> Label assignment 主要是指检测算法在训练阶段，如何给特征图上的每个位置进行**合适的学习目标的表示**，以及如何**进行正负样本的分配**。

 然而，在实际的目标检测任务中，每张图可能有多个 GTBox，网络的输出通常也是稠密的，Label Assignment 的过程会更为复杂，甚至可以说是检测网络训练的最核心的问题之一。如何科学地建立 N 个 GTBox 与 M 个网络预测值（包含分类、Box 回归等）的对应关系，来保证网络预测值拿到合理正负样本的 Label 去计算 Loss，从而用来高效的训练检测网络，这便是我们今天要讨论的问题。

**Label Assignment 发展过程**

**Label Assignment 的背景知识**

![6e3c9092838ccd0d0834b49972311946.png](https://img-blog.csdnimg.cn/img_convert/6e3c9092838ccd0d0834b49972311946.png)

图：RetinaNet PPL， 在测试过程中，Anchor 和网络预测叠加生成最后的检测框。

首先我们以检测器 RetinaNet 为例。网络包括 Backbone、Neck 和 Head 三部分，BackBone 提取特征；Neck 进行不同尺度的特征融合；Head 最终实现回归和分类的任务。在训练的过程中，假设网络的输出有 M=$W*H*A$ 个预测框以及相应得分。

![632438ee84ba22c5e71d8668b39f4982.png](https://img-blog.csdnimg.cn/img_convert/632438ee84ba22c5e71d8668b39f4982.png)

如上图所示，黄色以及红色框是人工标注的 GT，**黑色虚线分割出来的可以看作每一个待分配的样本** (在某些复杂情况下，每个黑色虚框存在多个待分配的样本)。那么如何合理地分配样本，例如，将图中 6 个蓝绿框都用来学习推车这个目标还是选择部分，其他的黑色虚框生成的相同蓝绿框这类先验 Anchor 又怎么分配。如何将人工标注框合理分配，这个过程就是我们今天要讨论的 Label Assginment。

图：黄框、红框为人工标注框，黑色虚框为待分配的样本。每个黑色虚框会产生类似于图中蓝绿框的先验 Anchor 框，蓝绿色为预设的正样本。