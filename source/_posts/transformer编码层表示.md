---
type: blog
title: Transformer的每一个编码层都学到了什么？
top: false
cover: true
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2019-09-18 17:32:47
password:
summary:
tags: Transformer
categories: NLP
---

*Transformer*现在已经被广泛应用于NLP领域的各项任务中，并且都取得了非常好的效果。其核心层使用了自注意力机制，关于为什么使用自注意力机制，作者提出了三点原因：

<!-- more -->

- 计算复杂度：*Transformer*的计算复杂度比*RNN*和*CNN*都要低
- 并行计算：*Transformer*可以进行并行计算，这也是作者提出*Transformer*模型的初衷
- 远距离长程依赖的路径距离：*Transformer*有更短的路径距离，因此更容易学习到远程的依赖关系。

前两个原因我们不做过多的介绍，只要仔细思考就可以理解，而且这两个是属于确定性问题，是可以通过理论分析得出的结论。但是第三点却是*Transformer*有效性的决定因素，而且无法进行理论分析（现在深度学习中的模型可解释性仍然是个研究热点），只有通过实验进行分析。本文就通过解读[An Analysis of Encoder Representations in Transformer-Based Machine Translation](https://www.aclweb.org/anthology/W18-5431)这篇论文来看下*Transformer*作者提出的第三点原因是否成立，并且深入理解*Transformer*每一层注意力都学到了什么。

本文通过不同方法分析了*encoder*层的注意力权重：

- 可视化注意力权重
- 注意力权重的树结构生成
- 将*encoder*作为不同预测任务的输入
- 将其中一个*encoder*的知识迁移到另一个里面

在研究*Transformer*中的注意力之前先训练一个*Transformer*模型，表1列出了训练模型的数据，表2列出了每组数据的*bleu*值：

表1：训练样本统计

|                    | # Training sentences |
| ------------------ | -------------------- |
| English → Czech    | 51,391,404           |
| English → German   | 25,746,259           |
| English → Estonian | 1,064,658            |
| English → Finnish  | 2,986,131            |
| English → Russian  | 9,140,469            |
| English → Turkish  | 205,579              |
| English → Chinese  | 23,861,542           |

表2：BLEU值

|                    | newstest 2017 | newstest 2018 |
| ------------------ | ------------- | ------------- |
| English → Czech    | 18.11         | 17.36         |
| English → German   | 23.37         | 34.46         |
| English → Estonian | -             | 13.35         |
| English → Finnish  | 15.06         | 10.32         |
| English → Russian  | 21.30         | 18.96         |
| English → Turkish  | 6.93          | 6.22          |
| English → Chinese  | 23.10         | 23.75         |

# 1. 注意力权重可视化

注意力权重可视化应该是最直接的一种研究方法。*Transformer*的*encoder*里面包含了6层注意力，每层注意力有8个head，要把这些权重全部可视化出来比较困难，所以作者选择了一些具有高视觉可解释性的注意力权重。

通过可视化他们发现这些注意力权重有四种模式：

- 在初始注意力层中，每个词的注意力会在自身上；
- 注意力层数增加后，注意力会集中在前几个词
- 或者注意力集中在后几个词
- 最后注意力会集中在句子的末尾

![](https://img.vim-cn.com/a2/1daa1130c9fa37195edb3385b35563a33905fa.png)

假设输入句子是“there is also an economic motive.”，*Transformer*中典型的注意力权重分布。可以看到layer 0中注意力都在词本身，后面基层的注意力会在词的前后几个词上，再到最后一层所有词的注意力全部放在句子末尾。这就预示着*Transformer*的高层注意力试图发现词与词之间的长程依赖关系，而低层注意力试图在局部发现依赖关系，这一发现很好的印证了*Transformer*作者的预想。

# 2. 树结构生成

*Transformer*的权重矩阵可以看成是一个加权图：每个次都是一个节点，词与词之间的注意力是边。尽管模型没有任何生成树形结构的训练，但是我们可以使用这种图结构来抽取出一棵树来看看它是否能反应词之间的依赖关系。

作者在*CoNLL 2017 Share Task*的*English PUD treebank*上做实验。

![](https://img.vim-cn.com/9b/0edca00f20e5f0e9e855fb9c06481dd1042b76.png)

上图展示了*UAS F1-Score*，实验中作为对比作者使用*random baseline*的得分是10.1，从上图我们可以看到虽然注意力层没有训练用于生成树结构，但是每一层的得分都好于随机，这说明模型可以学习到一些语法关系。

基本上得分最高的层都集中在前三层，说明前三层主要用于学习句子的语法结构的。

但是作者也通过在注意力权重可视化中发现的规律作为基准，计算*UAS F1-Score*得到最好的分数为35.08，而我们可以从上表中看到，表中最好的结果并没有比基准高多少，并且之前我们的结论是高层注意力用于学习远距离的依赖关系，但是这里我们看到主要是低层注意力学习到了有效的语法结构，这说明*Transformer*很难有效地处理更加复杂和长程的依赖。

总的来说，对于语料更丰富的数据来说，模型更能学习到语法结构（对比*English-Turkish*和其他），但是当语料达到一定程度的时候，模型并不能学到更多的语法知识了。

# 3. 探索序列标注任务

作者通过四个序列标注任务对*encoder*进行研究：

- Part-of-Speech tagging （ the Universal Dependencies English Web Treebank v2.0数据集）
- Chunking（CoNLL2000 Chunking shared task数据集）
- Named Entity Recognition（ CoNLL2003 NER shared task数据集）
- Semantic tagging（Parallel Meaning Bank (PMB) for Semantic tagging数据集）

表3：各数据集统计结果

|       | #labels | #training sentences | #testing sentences | average sent.length |
| ----- | ------- | ------------------- | ------------------ | ------------------- |
| POS   | 17      | 12543               | 1000               | 21.2                |
| Chunk | 22      | 8042                | 2012               | 23.5                |
| NER   | 9       | 14987               | 3684               | 12.7                |
| SEM   | 80      | 62739               | 4351               | 6.4                 |

![](https://img.vim-cn.com/da/b0a2810894b50fd0501e6dd70c0edbd38ebb53.png)

上图展示了测试结果。

- 对于POS和CHUNK任务来说，最高的结果基本出现在前三层，说明前三层用于学习语法结构，这一结论与之前的结果相吻合。
- NER和SEM这种注重语义知识的任务， 在模型的高层注意力中表现的更好，说明高层注意力主要用于学习语义知识。
- 上表数据右侧展示的是句子长度匹配的错误率，从表中我们可以看出，前三层的句子长度错误率普遍低于高层的注意力层，这说明低层注意力不仅要学习语法结构，还会对句子长度进行编码， 但到了高层注意力，句子长度信息就会丢失。唯一例外的是SEM任务，但是我们看下训练集的数据统计列标可以发现，SEM数据集的句子普遍较短，对模型来说更容易预测。而模型低层注意力本身就是在学习短程依赖方面比较有优势，高层注意力处理远距离依赖的乏力可能和句子长度信息的丢失有一定关系。
- 对比BLEU值我们可以发现，BLEU值越高，在序列标注任务中的表现也越好，这说明*encoder*对句子语法的编码越好，翻译质量会越高。

# 4. Transfer Learning

为了进一步研究编码进注意力层的知识是否对少语料的情景有所帮助，作者做了两个实验：

- 使用English-German训练好的编码层，初始化English-Turkish编码层，并且允许编码层权重微调（TL1）
- 使用English-German训练好的编码层，初始化English-Turkish编码层，并且保持权重不变（TL2）

![](https://img.vim-cn.com/8b/e008811c89ae0e7792e9d59f0bfb70ffe1b0e8.png)

上图展示了实验的结果。从结果中我们可以看到，效果很明显。

# 5. 总结

本文主要研究了*Transformer*每层注意力都学到了什么，通过实验发现：

- 低层注意力能有效地获得句子的短程依赖关系
- 高层注意力尝试学习句子的远距离依赖关系，但是效果并不明显
- 低层注意力能够对句子长度进行编码，高层注意力可能丢失句子长度编码信息，这也可能是导致远距离依赖关系处理困难的原因
- 低层注意力主要学习句子语法结构
- 高层注意力主要学习句子语义信息
- 语料越多，模型越能学到足够多的语法信息，但是存在一个瓶颈
- 语法信息越丰富，翻译的效果越好
- 用大量语料学习到的知识可迁移到少量预料任务中去

# 6. 参考资料

[An Analysis of Encoder Representations in Transformer-Based Machine Translation](https://www.aclweb.org/anthology/W18-5431)

