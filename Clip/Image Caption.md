# Image Caption

**image caption 是指用自然语言描述图像中的视觉内容的任务，通常采用一个视觉理解系统和一个能够生成有意义的、语法正确的句子的语言模型**

Image caption 任务的目标是找到最有效的 pipeline 来处理输入图像，表示其内容，并通过在保持语言流畅性的同时生成视觉和文本元素之间的连接，将其转换为一组单词序列 [1](https://blog.csdn.net/Bit_Coders/article/details/119566024#fn1)。

## 数据集概览

早期的 image caption 主要采用 Flickr30K 和 Flickr8K 数据集，这个数据集图片来源于 Flickr 网站。

目前比较常用的数据集是 COCO Captions、Conceptual Captions (CC)，包含人、动物和普通日常物品之间的复杂场景的图像。

COCO Captions、Conceptual Captions (CC)、VizWiz、TextCaps、Fashion Captioning、CUB-200 等数据集的标注样例如下图（a）所示，数据集中语料库的高频词云如下图（b）所示 [1](https://blog.csdn.net/Bit_Coders/article/details/119566024#fn1)，可以反映数据集中主要目标类别的分布。

![在这里插入图片描述](https://img-blog.csdnimg.cn/88b773aca07e4af3a207da4fc7082386.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6Jm-57Gz5bCP6aaE6aWo,size_20,color_FFFFFF,t_70,g_se,x_16)

### 标注方式

COCO Captions、Conceptual Captions (CC) 数据集中对图像描述的标注，是基于**整幅图像**的。Flickr30K Entities 标注了 Flickr30K 中 caption 里提到的**名词**，并标注了对应的 bbox。Visual Genome 数据集提供了描述图像中区域的短语，并使用这些区域来生成一个场景图（scene graph）。Localized Narratives 为**每个单词** 都提供了基于其跟踪片段所表示的图像中的一个特定区域，包括名词、动词、形容词、介词等。[2](https://blog.csdn.net/Bit_Coders/article/details/119566024#fn2)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2cfff3baa58248f8a38dd672695cc235.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6Jm-57Gz5bCP6aaE6aWo,size_20,color_FFFFFF,t_70,g_se,x_16)

https://blog.csdn.net/Bit_Coders/article/details/119566024