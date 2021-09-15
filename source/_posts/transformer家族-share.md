---
type: blog
title: Transformer家族之Share Attention Networks
top: false
cover: true
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2019-09-30 14:36:44
password:
summary:
tags: [Transformer, NMT]
categories: [NLP]
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/5396ee05ly1g5pqn3ch6zj20u092znph.jpg)

接下来我们介绍一下[Sharing Attention Weights for Fast Transformer](https://arxiv.org/abs/1906.11024)这篇文章。实际上这篇文章之前还有两个关于加速*Transformer*推理的文章，一个类似*Latent Transformer*的引入离散隐变量的类*VAE*的方法，另一个是引入语法结构达到*non-auto regression*的方法。个人感觉没什么意思，就直接跳过了。本文要介绍的这篇文章比较有意思，引入共享权重的概念。从近期关于*Bert*小型化的研究（比如DistilBERT，ALBERT，TinyBERT等）来看，实际上*Transformer*中存在着大量的冗余信息，共享权重的方法应该算是剔除冗余信息的一种有效的手段，因此这篇文章还是比较有意思的。

<!--more-->

# 1. Attention Weights

我们在介绍*Transformer*的时候说过，*Multi-Head Attention*其实和*Multi-dimension Attention*是一回事。在多维注意力机制中，我们希望每一维注意力都能学到不同的含义，但是实际上[Lin et al. 2017](https://openreview.net/pdf?id=BJC_jUqxe)研究发现，多维注意力机制经常会出现多个维度学到的东西是相同的的情况，即不同维度的注意力权重分布是相似的。

本文做了一个类似的研究，发现不同层之间的注意力权重也有可能具有相似的分布，说明不同层之间的注意力权重也在学习相同的信息。作者分别计算了不同层之间的注意力权重的JS散度，用以说明不同注意力权重有多大的差异性，下图是作者的计算结果：

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/457db2eeb701029cdb46fb1d801e9af3448fbc.png)

注意作者的意图是加速*Transformer*的推理速度，因此着重研究的是*decoder*部分，因此上图表示的*decoder*的JS散度矩阵。我们知道每个*decoder block*中包含两个注意力矩阵，一个是*masked atteniton*，一个是*encoder-decoder attention*，上图左图表示*masked attnetion*，右图表示*encoder-decoder attention*。图中颜色越深表示差异性越小，即两个注意力权重矩阵越相似。从图中可以看到*self-attention*部分的相似非常大，而*encoder-decoder attention*相似性虽然不如*self-attention*，但是1,2,3,4之间和5,6层之间的相似性也比较大。

由于不同层的注意力权重相似性较大，因此可以在不同层中共享同一个注意力权重，减少参数从而达到加速推理的目的。

> 这里作者只讨论了*decoder*的情况，还记得我们之前介绍过一篇论文[An Analysis of Encoder Representations in Transformer-Based Machine Translation](https://www.aclweb.org/anthology/W18-5431)，讲的是在*encoder*的每一层注意力都学到了什么信息。这里我希望对比一下用本文的方法和上面的研究方法对比一下，看看得到结论是否能相互印证。因此我和本文的作者进行了交流，很遗憾的是作者并没有像*decoder*那样仔细研究*encoder*的情况，但是作者认为理论上*encoder*的情况应该是和*decoder*的*self-attention*的情况是一致的，因为*encoder*中的*attention*和*decoder*的*self-attention*层都是对单一的序列进行编码，不同的是前者是对源序列，后者是对目标序列，因此两者应该有相似的表现。虽然作者没有计算*encoder*注意力权重的JS散度，但是在实验过程中，尝试过对*encoder*的注意力进行共享，发现第1, 2层计算，3-6层使用第1层的权重，此种情况下对性能也未发现明显的下降趋势，因此作者认为*encoder*端不同层的注意力权重同样存在着较多的相似情况。

# 2. 模型结构

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/bd0042560fedcfb699684b22b00f4d2f2d96b0.png)

模型结构如上图。其实*SAN*的基本想法很简单，就是计算一次注意力权重，多次重复使用。具体的数学描述如下：

- **Self-Attention**

定义第$m$层的注意力自注意力权重矩阵为：
$$
S^m = s(Q^m, K^m)
$$
有了第$m$层的注意力权重，把它共享给第$m+1$层：
$$
S^{m+i} = s(Q^m, K^m), i \in [1, \pi-1]
$$
其中$\pi$表示有多少层共享第$m$层的注意力权重，比如对于一个6层的*decoder*来说，我们可以让前两层共享一个权重矩阵$\pi_1=2$，让后面的4层共享两一个权重矩阵$\pi_2=4$，具体的共享策略在后面介绍。

- **Encoder-Decoder Attention**

对于*encoder-decoder*注意力层来说，采取同样的操作。但是为了进一步加速推理，这里作者采用了一个小技巧，即$K, V$使用*encoder*的输出：
$$
A^{m+i} = A^m = S^m \cdot V, i \in [1, \pi-1]
$$
其中$A^m$是第$m$层的 注意力输出，$V$是*encoder*的输出。

另外为了减少内存消耗，共享的注意力权重只需要拷贝给相应的层即可，不需要配置到在每一层的内存中去。

# 3. Learning to Share

那么，现在的问题是，我们怎么知道那些层需要共享注意力权重呢？一个最简单的方法就是遍历测试，然后在*development set*上进行微调。显然这不是最优的方法，因为我们要在不同的程度上控制注意力权重的共享程度。

这里作者提出采用动态策略——利用JS散度计算层与层之间的相似度：
$$
\mathrm{sim}(m,n) = \frac{\sum_{i=m}^n \sum_{j=m}^n(1-\delta(i,j))\mu(i, j)}{(n-m+1)\cdot (n-m)}
$$
其中$\mu(i, j)$表示第$i$层和第$j$层的JS相似度，$\delta(i,j)$表示*Kronecker delta function*。上式表示$m,n$层之间的相似性，当$\mathrm{sim}(m,n) > \theta$时那么$m,n $层就共享注意力权重。

首先从第一层开始计算满足$\theta$阈值的最大的$\pi_n$，如此往复，直到所有的注意力层都计算完了。这个时候我们会得到一个注意力权重的共享策略$\{\pi_1, ..., \pi_N\}$，$\pi_i$实际上前面已经解释了表示什么，但是为了更直观的解释，这里我们还是举个例子吧：

假设*deocder*有6层注意力层，共享策略为$\{\pi_1=2,\pi_2=4\}$，表示第$1,2$层共享注意力权重，第$3,4,5,6$共享注意力权重。

一旦共享策略确定了下来之后，我们要重新训练模型，对注意力权重进行微调，直到模型收敛。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/56794343674c0e10d5472a191f55512a193361.png)

# 4. 实验结果

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/c7ed7dba2280159161a92a1c0a7de50419dda7.png)

# 5. 参考资料

[Sharing Attention Weights for Fast Transformer](https://arxiv.org/pdf/1906.11024.pdf) Tong Xiao et al., 2019