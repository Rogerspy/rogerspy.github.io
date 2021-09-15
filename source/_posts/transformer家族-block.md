---
type: blog
title: Transformer家族之Blockwise Transformer
top: false
cover: True
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2020-03-26 15:35:28
password:
summary:
tags: [Transformer, NMT]
categories: [NLP]
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/5396ee05ly1g5pqn3ch6zj20u092znph.jpg)

本文将继续介绍关于*transformer*在 *non-auto regression*方面的研究，今天要介绍的是*Google Brain*的工作[Blockwise Parallel Decoding for Deep Autoregressive Models](https://papers.nips.cc/paper/8212-blockwise-parallel-decoding-for-deep-autoregressive-models.pdf)。

<!-- more -->

# 1. 简介

对于*seq2seq*任务来说，*decoding*过程是影响推理速度的关键。效率和效果似乎是一个难以和平共处的老对手，纵观之前我们介绍的方法，大致分成三类：*auto regression*、*semi-auto regression*、*non-auto regression*。*Auto regression*是效果最好的，但是推理速度是最慢的；*non-regression*是推理速度最快的（可以比前者快甚至几十倍），但是效果就差了；而*semi-auto regression*介于两者之间，效果略差但速度较快。之前我们介绍的*Share transformer*在效率和效果上都有所提升，但是这种提升是来源于对模型结构的优化，以更小的计算量实现的，虽然我很喜欢，但是有没有一种更新的，在*decoding*上的创新方法，既能提升效果，又能提高效率呢？下面要介绍的这篇文章就完成了这样一个设想。



# 2.Blockwise Parallel Decoding

![](https://img.vim-cn.com/76/437388cec32870c4ad92e3128071830643baac.png)



上面展示的是*BPD*的基本方法。

假设我们有一组模型$\{p_1, p_2, ..., p_k\}$，其中$p_1$是 原始模型，$p_2, ..., p_k$是辅助模型。从图中我们可以看到，推理过程分成三步：

- **Predict**: 使用 $p_1$ 预测一个长度为$k$的序列；

$$
\hat{y}_{j+i}=\arg max_{y_{j+i}} p_i(j_{j+i}| \hat{y}_{\le j}, x), ~~ i=1,2, ..., k
$$

- **Verify**: 使用辅助模型找到$i$的最大值$\hat{k}$，使得

$$
\hat{y}_{j+i}=\arg max_{y_{j+i}} p_1(j_{j+i}| \hat{y}_{\le j+i-1}, x), ~~ 1 \le i \le k
$$

- **Accept**: 扩展预测序列，从$\hat{y}_{\le j}$变成$\hat{y}_{\le j+\hat{k}}$，同时设置$j \leftarrow j +\hat{k}$。



> 在读论文的时候感觉作者在对*predict*和*verify*过程进行详细描述的时候给搞反了，不知道是真的写反了还是我理解错了，已经给作者发邮件了，截止到现在作者还没有回我。这里暂时按照我的理解做介绍，如果是我理解错了，再做修改。



假设我们现在已经有一个种子序列：
$$
\mathrm{I\quad saw\quad a\quad dog\quad ride}
$$

- 预测阶段：将种子序列作为$p_1$的输入，使$p_1$并行输出一个候选序列： $\rm{in \quad the\quad bus}$

- 验证阶段：

  ① 将 $\mathrm{I \quad saw \quad a \quad dog \quad ride}$作为$p_1$的输入去预测 $\rm{in}$

  ② 将 $\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in}$作为$p_2$的输入去预测 $\rm{the}$

  ③ 将 $\mathrm{I \quad saw \quad a \quad dog \quad ride}$作为$p_3$的输入去预测 $\rm{bus}$

![](https://img.vim-cn.com/f2/a7da5078c2becf83362e61ed60abc8bf20b98a.png)

$p_1$模型的输入来源无需多说，就是原始的种子序列；$p_2$的输入是种子序列+在预测阶段$p_1$的输出的第一个词；$p_3$的输入是种子序列+预测阶段$p_1$的输出的前两个词，以此类推。在验证阶段每个模型都是以*Greedy decoding*模式进行推理的，即只预测下一个词。因为每个模型都是独立运行的，所以$p_1, p_2, p_3$是并行运算的。

- 验收阶段：

  在前面的验证阶段我们的到了一个候选集：

$$
\rm{I\quad saw\quad a\quad dog\quad ride\quad in} \\\\
\rm{I\quad saw\quad a\quad dog\quad ride\quad in\quad the} \\\\
\rm{I\quad saw\quad a\quad dog\quad ride\quad in\quad the\quad car}
$$

与预测阶段$p_1$的输出对比发现$p_1$的输出最后一个词是$\rm{bus}$，而$p_3$输出的词是$\rm{car}$，相当于$p_1$和$p_3$发生了冲突，那么在验收阶段我们只取验证阶段与预测阶段输出相同的结果的最长序列，即我们取$\rm{I\quad saw\quad a\quad dog\quad ride\quad in \quad the}$作为这一轮预测的输出，同时也是下一轮的输入。

相比于传统的*auto-regression*每一轮预测只能生成一个词，这种*Blockwise decoding*每一轮可以产生多个词（上面的例子中，同时生成$\rm{in\quad the}$两个词，当然需要注意的是每一轮生成词序的长度并不是固定的），同时还能保证这些词是以*auto-regression*的方式产生的，因此在提升推理效率的同时，效果一定不会比*auto-regression*差。

这里再唠叨几句，为什么这种方法是合理的呢？因为我们先用$p_1$生成一个短序列，然后用$p_1, p_2, ...,p_k$去验证的时候，相当于是将一个时序的推理过程并行化了。首先，我们先假设$p_1$生成的序列是合理的，在验证过程中用第一个模型去验证如果是在自回归模式下$p_1$是否会生成 $\rm{in}$，用第二个模型验证自回归模式下$p_1$ 是否会在$\rm{in}$后面生成$\rm{the}$，以此类推，知道我们发现验证阶段的输出和验证阶段的输出不一致的时候，我们就认为并行化生成的序列在这个位置及以后的输出是不合理的，所以我们可以放心的使用之前的生成的序列。

前面我们是以已经存在一个种子序列作为初始条件介绍的，那么在模型推理的一开始还没有种子序列的时候其实也差不多。*Auto-regression*的时候第一步推理也是根据*encoding*生成第一个词，然后逐步生成我们想要的序列。在*blockwise decoding*的时候也是一样的，初始的时候$p_1$不是生成一个词，而是生成一个序列。这样我们就可以重复上面的三个步骤了。

文章中给出了一个实际的推理过程的例子：

![](https://img.vim-cn.com/60/413eb68aeb9dd9c4f24c597d73c1d1a4eaec0d.png)

另外，还有一个需要注意的细节，在文章当中并没有提及，我不太清楚作者是怎么处理的，这里我说下我个人的想法。还以上面的句子为例：初始序列是: $\mathrm{I \quad saw \quad a \quad dog \quad ride}$

假设预测阶段$p_1$生成的序列是$\rm{in \quad the \quad bus}$，但是在验证阶段$p_1$生成的是$\rm{on}$，也就是说，验证阶段的第一个模型输出就与预测阶段生成的第一个词不一致，这种情况该怎么办呢？当然需要指出的是，这种情况的概率应该是很低的，因为都是$p_1$模型的输出所以在最靠近输入序列的输出应该有相同的分布，但是在写代码的时候不得不考虑这种情况。我个人的想法是应该是以验证阶段$p_1$的输出为准，毕竟这个时候的模型推理模式是*auto-regression*式的。虽然文章没有确切提出这一处理方式，但是在下面的介绍中，我们可以看到作者实际上确实是采用了这种方法的。



# 3. Combined Scoring and Proposal Model

从上面的推理步骤可以看出，每一步推理都需要至少两次模型调用：预测阶段调用$p_1$和验证阶段调用$p_1, p_2, ..., p_k$。理想情况下，我们希望生成长度为$m$的序列，只调用$m/k$次模型，但是上面的步骤需要$2m/k$次模型调用。那么有没有什么办法尽可能接近$m/k$次呢？接下来我们就介绍对上面步骤的改进，使得调用次数从$2m/k$次下降到$m/k+1$次。

![](https://img.vim-cn.com/d0/b5e3d64b7e1e913745c7882eefe7361619caae.png)



和之前一样预测阶段先由$p_1$生成一个短序列，但是在验证阶段就发生了变化：之前是每个模型生成一个词，现在我们令每个模型和预测阶段的$p_1$一样，都生成一个短序列，这个实际上就相当于是验证我那个阶段和预测阶段合二为一了。我们仍然通过对比生成序列的第一个词来进行验收，而下一个推理周期的预测序列采用刚刚验收的序列生成的短序列。

以上图为例：

第一步：使$p_1$并行输出一个候选序列： $\rm{in \quad the\quad bus}$

第二步：

① 将 $\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in }$作为$p_1$的输入去预测 $\rm{the \quad car \quad last}$

② 将 $\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the }$作为$p_2$的输入去预测 $\rm{car \quad this \quad week}$

③ 将 $\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the \quad bus}$作为$p_3$的输入去预测 $\rm{last \quad week \quad when}$

第三步：我们提取出 $\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the }$作为我们这一轮的推理的输出，同时将$\rm{car \quad this \quad week}$作为下一轮推理的预测序列。这样我们就相当于每一轮推理只需要调用一次模型（时间）就完成了验证和预测两步。

我们可以看到，这种推理方法的核心点在于$p_1$必须具有并行生成序列的能力，不能是*RNN*一次生成一个词的模型，所以只能是*CNN*或者*Transformer*这种具有并行能力的模型。

# 4. Approximate Inference

到目前为止，我们就介绍了*BPD*的基本方法，实际上我们还可以放松验证标准实现额外的加速推理。



## 4.1 Top-$k$ Selection

 之前在验证阶段，我们要求辅助模型的输出必须和原始模型的输出高度一致才可以，但是实际上我们可以认为，只要原始模型的输出在辅助模型的输出的*top-k*个输出中就可以，即
$$
\hat{y}_{j+i} \in \mathrm{top}k ( p_1(y_{j+i}|\hat{y}_{\le j+i-1}, x))
$$
比如前面$\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the \quad car  \quad this \quad week}$ 其中 $\rm{car}$ 与 $\rm{bus}$ 不一致，所以我们的输出序列只取到$\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the}$。那么现在如果 $\rm{car}$ 存在于输出概率的*top-k*候选里面，我们也认为它是正确的，那么我们这一轮的推理输出结果应该是$\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the \quad car}$。

## 4.2 Distance-Based Selection

如果两个词在语义空间中有相近的含义，那么我们也可以用近义词代替，即：
$$
d(\hat{y}_{j+i}, \arg max_{y_{j+i}} p_1(y_{j+i}|\hat{y}_{j+i-1}, x)) \le \epsilon
$$
比如 $\rm{car}$ 和 $\rm{bus}$ 的意思相近，

因此我们任然可以得到 $\mathrm{I \quad saw \quad a \quad dog \quad ride \quad in \quad the \quad car}$。

## 4.3 Minimum Block Size

在推理的时候有可能出现炎症阶段第一个词就不正确的情况（上面我们讨论过），这样的话我们就只能一次生成一个词，那么久退回到自回归的推理模式了，为了保证推理的加速，我们可以设置一个最小值$l$，$1\le l \le k$，无论对错，每次最小生成 $l$ 个词。当$l=1$时，模型退化到自回归；当 $l=k$ 时模型变成纯并行模型。



# 5. Traning

从上面描述我们可以看到，这种推理方式是非常消耗内存的，因为需要同时存在多个模型，这样我们就不能计算全部模型的loss了，取而代之，我们可以随机选择其中一个模型的loss作为整体的loss。



# 6. Experiments

![](https://img.vim-cn.com/99/d5f7574dcd44e5f528f917130b2e50cdd20740.png)



# 7. Reference

*Mitchell Stern, Noam Shazeer, Jakob Uszkoreit*, NeurIPS 2018, [Blockwise Parallel Decoding for Deep Autoregressive Models.](https://papers.nips.cc/paper/8212-blockwise-parallel-decoding-for-deep-autoregressive-models.pdf)

