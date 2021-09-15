---
type: blog
title: Transformer家族之Average Attention Network
top: false
cover: true
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2019-09-24 09:54:06
password:
summary:
tags: [Transformer, NMT]
categories: [NLP]
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/5396ee05ly1g5pqn3ch6zj20u092znph.jpg)

*Transformer*虽然在训练上比*RNN*和*CNN*快，但是在做推理（*decoding*）的时候由于采用的是*Auto-regression*不能做到并行计算，所以速度很慢（甚至可能比纯*RNN*还要慢），所以针对这种情况很多研究者提出了*decoding*时也能采用并行计算的改进方案，下面要介绍的这个*transformer*大家族的以为成员就是其中之一：**Average Attention Network**。

<!--more-->

# 1. Average Attention Network

这篇论文作者认为造成*transformer*在decoding的过程慢的原因有两个：

- Auto-regression
- self-attention

是的，作者认为在*decoder*中，对目标句子编码时计算目标句子内部依赖关系的时候使用自注意力机制会造成解码速慢，因此作者提出使用*AAN*进行代替。

## 1.1 模型结构

![](https://img.vim-cn.com/04/c4f0be76f3145e451270ec8d3e0fa91f7c5a4c.png)

从模型结构图中可以看到，基本结构和*Transformer*基本相同，唯一不同的点在于*decoder layer*中*Masked Multi-head attention*变成了*Average Attention*。那么下面我们就来看看这个*Average Attention*是何方神圣？

## 1.2 Average Attention

![](https://img.vim-cn.com/fd/334ec556af95178141cbbc11f4f0b63124c3f7.png)

结构如图所示，给定输入$\mathbf{y = \{y_1, y_2, ..., y_m\}}$

- *AAN*首先计算累加平均：

$$
\overline{\mathbf{y}_j} = \frac{1}{j} \sum_{k=1}^j \mathbf{y}_k
$$

假设模型auto-regression产生了$j=3$个词，“我，喜欢， 打”，对其中每个词进行累加平均得（average(我), average(我+喜欢), average(我+喜欢+打)）。

- 然后经过*FFN*层进行线性变换，其中*FFN*层即*Point wise feeed forward*：

$$
\mathbf{g}_j = \mathrm{FFN}(\overline{\mathbf{y}_j})
$$

这两部虽然很简单，但是却是*AAN*中最核心的部分:

1. 每个位置上的向量都是通过之前的词计算得来，所以词与词之间并非独立的，而是存在依赖关系的；
2. 无论输入向量有多长，之前的词都被融合进同一个向量中，也就是说词与词之间的距离是不变的，这就保证了*AAN*可以获得长距离依赖关系。

> 注意：在作者提供的源码中，FFN层是可以跳过的，即计算出平均值以后不经过FFN层，直接进行接下来的计算。

- 拼接原始的输入和累加平衡后的输出

$$
c = \mathrm{Concat}(\mathbf{y}_j, \mathbf{g}_j)
$$

- 由于后面加入了*LSTM*的遗忘门机制， 因此这里先计算各个信息流所占的比例：

$$
\mathbf{i}_j, \mathbf{f}_j = \sigma(Wc)
$$

其中，$\mathbf{i}_j$表示原始输入在接下来的信息流中所占的比例，$\mathbf{f}_j$表示在接下来的信息流中*Average*所占的比重。

- 遗忘门

$$
\widetilde{\mathbf{h}}_j = \mathbf{i}_j \odot \mathbf{y}_j + \mathbf{f}_j \odot \mathbf{g}_j
$$

- 残差连接

$$
\mathbf{h}_j = \mathrm{LayerNorm}(\mathbf{y}_j + \widetilde{\mathbf{h}}_j)
$$

至此，整个*AAN*就计算完成了，这也是*original AAN*。

## 1.2 Masked ANN

之前我们介绍*origin AAN*时说到求累加平均的时候举了个例子，假设我们想要预测“我非常喜欢打篮球”， 在auto-regression时：

输入：$(我, 非常, 喜欢, 打)$

计算累加平均的时候要分别计算：average(我)，average(非常)，average（喜欢），average(打)。也就是说我们要多次计算平均值，这样就不能实现并行化计算，所以我们希望能通过矩阵一次性求出这三个平均向量：
$$
\left\{
\begin{array}
{cccc}
1 & 0 & 0 & 0\\\\
1/2 & 1/2 & 0 & 0 \\\\
1/3 & 1/3 & 1/3 & 0 \\\\
1/4 & 1/4 & 1/4 & 1/4
\end{array}
\right\} \times 
\left( 
\begin{array}
{c}
y_1 \\\\
y_2 \\\\
y_3 \\\\
y_4
\end{array}
\right) =
\left( 
\begin{array}
{c}
y_1 \\\\
\frac{y_1+y_2}{2} \\\\
\frac{y_1+y_2+y_3}{3} \\\\
\frac{y_1+y_2+y_3+y_4}{4}
\end{array}
\right)
$$

##  1.3 Decoding Acceleration

不同于*transformer*中的自注意力机制，*AAN*可以以非常快的速度进行推理：
$$
\widetilde{\mathbf{g}}_j = \widetilde{\mathbf{g}}_{j-1} + \mathbf{y}_j \\\\
\mathbf{g}_j = \mathrm{FFN}(\frac{\widetilde{\mathbf{g}}_j}{j})
$$

即，在进行*auto-regression*的时候每次都只需要用之前的结果与本次的输入相加，然后求平均即可。注意$\widetilde{\mathbf{g}}_0=0$。我们只需要根据前一次的状态就可以确定当前的状态，而不需要像自注意力机制那样依赖之前所有的状态。

至此关于*AAN*的部分我们就介绍完了，模型其他部分的结构和*transformer*保持一致，在作者的实验中，虽然*BLEU*值较*transformer*略微降低，但是推理效率上提升很大， 尤其是对长句。

# 2. 实验结果

![](https://img.vim-cn.com/2e/bec247df017d256ee656ca350cf04e57aeb2e2.png)

# 3. 核心代码

## 3.1 pytorch

```python
class AAN(nn.Modules):
    def __init__(...):
        super(AAN, self).__init__()
        
    def forward(...):
        pass
```

## 3.2 tensorflow

```python
class AAN(tf.keras.layers.Layer):
    def __init__(...):
        super(AAN, self).__init__()
        
    def call(...):
        pass
```



# 4. 参考资料

[Accelerating Neural Transformer via an Average Attention Network](https://arxiv.org/pdf/1805.00631.pdf)



