---
type: blog
title: 预训练语言模型：Pre-trained seq2seq
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-09-17 15:57:24
password:
summary:
tags: [Language Model, pre-trained seq2seq]
categories: 语言模型
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/151034571.png)

之前我们介绍过 [seq2seq 模型](https://rogerspy.github.io/2019/08/26/NLP%E4%B8%AD%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%AE%80%E4%BB%8B%EF%BC%88%E4%B8%80%EF%BC%89/)，通常用作机器翻译，通过编码器（encoder）对源语言进行编码，然后通过解码器（decoder）对编码器的结果进行解码，得到目标语言。原始的 seq2seq 模型是使用平行语料对模型从头开始进行训练，这种训练方式需要大量的平行语料。[Prajit Ramachandran](https://arxiv.org/pdf/1611.02683.pdf) 提出一种方法，可以大幅降低平行语料的需求量：先分别使用源语言和目标语言预训练两个语言模型，然后将语言模型的权重用来分别初始化编码器和解码器，最终取得了 SOTA 的结果。

<!--more-->

# 1. Method

## 1.1 Basic Procedure

给定输入序列 $x_1, x_2, ..., x_m$，seq2seq 的目的是最大化：
$$
p(y_n, y_{n-1}, ..., y_1|x_1, x_2,...x_m) = \prod_{t=1}^n p(y_t|y_{t-1},...,y_1;x_1, x_2,...,x_m)
$$
seq2seq 模型是使用编码器（RNN）将 $x_1, x_2, ..., x_m$ 表示成一个隐向量，然后将隐向量传递给解码器进行序列解码。我们的方法是将编码器和解码器都当做 RNN 语言模型进行使用大量的语料进行预训练。

<img src="https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20220114180418.png" style="zoom:80%;" />

两个语言模型训练完成以后，将两个语言模型的权重用来初始化编码器和解码器。为了方便起见，解码器的 $\text{softmax}$ 使用目标语言的语言模型的 $\text{softmax}$ 进行初始化。

## 1.2 Monolingual language modeling losses

使用语言模型初始化 seq2seq 以后，再用平行语料进行 fine-tuning。根据 [Goodfellow et al. 2013](https://arxiv.org/pdf/1312.6211.pdf) 的研究，fine-tuning 过程很容易造成灾难性遗忘（catastrophic forgetting），使得模型在语言模型上的性能急剧下降，损害模型的泛化能力。

为了保证模型在平行语料上不会过拟合，在fine-tuning 阶段继续训练语言模型任务，seq2seq 和 语言模型任务的损失等权相加作为最终损失。

## 1.3 Other improvements to the model

预训练和损失叠加机制能大幅提升模型性能，但是我们发现另外两个可以小幅提升模型能力的技巧：

1. 残差连接；
2. 多层注意力。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20220114214211.png)

# 2. Experiments

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20220114215019.png)

# Reference

1. [Unsupervised Pretraining for Sequence to Sequence Learning](https://arxiv.org/pdf/1611.02683.pdf), *Prajit Ramachandran, Peter J. Liu and Quoc V. Le* 2017, arxiv: 1611.02683
2. [An empirical investigation of catastrophic forgetting in gradient-based neural networks](https://arxiv.org/pdf/1312.6211.pdf), *Ian J Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. 2013. arXiv preprint arXiv:1312.6211*



