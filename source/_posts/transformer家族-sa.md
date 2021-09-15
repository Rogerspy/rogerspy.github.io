---
type: blog
title: Transformer家族之Semi-Autoregressive Transformer
top: false
cover: true
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2019-09-30 09:49:38
password:
summary:
tags: [Transformer, NMT]
categories: [NLP]
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/5396ee05ly1g5pqn3ch6zj20u092znph.jpg)

从题目就可以看出来，本文将要介绍一种半自动回归的机器翻译解码技术。之前我们介绍了各种非自动回归的解码技术，其中有一个*Latent Transformer*是先用*auto-regression*生成一个短序列，然后用这个短序列并行生成目标序列。当时我们说这其实算是一种半自动回归的方法。今天我们要介绍另一种半自动回归的方法——*SAT*。

<!--more-->

和*Latent Transformer*不同，*Semi-Auto regressive Transformer （SAT）*仍然是*auto-regression*的运行机制，但是不像*auto-regression*那样一次生成一个元素，*SAT*是一次生成多个元素。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/8dda2a1d508a3e3108c09a543a4a4c918c2e76.png)

# 1. 模型结构

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/f6757519d2f1b55a549102a43e27fbc55872ea.png)

模型结构如图。我们可以看到模型基本上和*Transformer*保持一致，只在图中红虚线框中有所改变。

## 1.1 Group-Level Chain Rule

给定一个目标序列，通常的建模方式是词级别的链式法则：
$$
p(y_1, y_2, ..., y_n|X) = \Pi_{t=1}^n p(y_t|y_1, ..., y_{t-1}, x)
$$
每个词都依赖之前的词。在*SAT*中，对序列进行分组
$$
G_1, G_2, ..., G_{[(n-1)/K]+1} = y_1, ..., y_K, y_{K+1}, ..., y_{2K}, ...,y_{[(n-1/K)\times K+1]}, ..., y_n
$$
其中$K$表示每组的大小，$K$越大 表示并行能力越强，除了最后一组其他每组中必须包含$K$个词。这样的话，上面的链式法则则变成：
$$
p(y_1, ..., y_n|X) = \Pi_{t=1}^{[(n-1)/K]+1} p (G_t|G_1, ..., G_{t-1}, x)
$$
作者将之前的*auto-regression*称之为*short-distance prediction*，而将*SAT*称之为*long-distance prediction*。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/30cac6ad33a85cd226d29a38b3f3a70fe8a773.png)

## 1.2 Relaxed Causal Mask

在*Transformer*中由于是连续的词矩阵，因此在做*mask*的时候直接使用下三角矩阵，但是在*SAT*中由于答对词进行了分组，再使用下三角矩阵就不合适了，作者这里提出一个**粗粒度下三角矩阵**（*coarse-grained lower triangular matrix*）的 *mask*矩阵。如图所示：

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/0c0a6933a204fd767ede1e4794779d4f51378f.png)

图中左边表示标准的下三角矩阵，右边表示粗粒度下三角矩阵。

数学形式：

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/49d1f6c42a21bfff8b118dbefe0d5cd004a2ea.png)

# 2. 实验结果

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/42129a015c5e78016cab0e78fc549d6d48afce.png)

# 3. 参考资料

[Semi-Autoregressive Neural Machine Translation](https://arxiv.org/pdf/1808.08583.pdf)

