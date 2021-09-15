---
type: blog
title: 预训练语言模型-对数线性语言模型
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-03-16 23:46:29
password:
summary:
tags: [NLP, Log-Linear Language Model]
categories: 语言模型
---

# 1. 前言

回想语言模型问题，我们的任务是在给定前 $j-1$ 个词的情况下，预测第 $j$ 个词：
$$
p(W_j=w_j|W_1=w_1, W_2=w_2, ..., W_{j-1}=w_{j-1}) =  p(w_j|w_1, w_2, ..., w_{j-1})
$$
在马尔科夫假设条件下：
$$
p(w_j|w_1, w_2, ..., w_{j-1}) \approx  p(w_j|w_{j-n+1:j-1})
$$
<!--more-->

在进行参数估计的时候，为了避免因数据稀疏化造成的零词频问题，引入了各种平滑技术。在使用平滑技术的时候，有诸多限制，比如稍有不慎可能造成概率和不为 1 的情况，比如对于线性插值法来说，一旦 n-gram 过多引入的 $\pmb{\alpha}$ 超参数的搜索空间也会变大，找到一个合适的 $\pmb{\alpha}$ 会变得很困难，另外参数估计的复杂度也会变得很高；对于加性平滑和减值平滑来说，对于零词频词的概率分配引入过多的人为假设。

下面我们介绍一种新的解决数据稀疏带来的参数估计困难的方法——对数线性语言模型（log-linear language model）。实际上对数线性模型在自然语言处理领域有着广泛的应用，这里只介绍利用对数线性模型对语言进行建模，即所谓的对数线性语言模型。

# 2. 对数线性模型（Log-Linear Model）

假设有一个模型可能的输入集 $\mathcal{X}$ ，一个模型可能的有限输出集 $\mathcal{Y}$，一个函数 $f:\mathcal{X}\times\mathcal{Y}\rightarrow\mathbb{R}^d$ 可以将 $(x,y)$ 映射到一个特征向量，其中 $d$ 为向量的维度，另外还有一个参数向量 $v\in\mathbb{R}^d$。对于任意的 $\{(x, y)|x\in \mathcal{X}, y\in \mathcal{Y}\}$，我们定义：
$$
p(y|x; v) = \frac{\exp(v\cdot f(x, y))}{\sum_{y' \in \mathcal{Y}}\exp(v \cdot f(x, y'))}
$$
其中 $v \cdot f(x,y) = \sum_{k=1}^d v_k \cdot f_k(x, y)$ 表示 $v$ 和 $f(x, y)$ 的内积。上式的意义是在模型参数为 $v$ 时，在 $x$ 的条件下，出现 $y$ 的概率。

既然模型的名字是线性对数模型，那么显然，我们要对上式求对数：
$$
\log p(y|x;v) = v \cdot f(x,y) -\log \sum_{y' \in \mathcal{Y}} \exp(v \cdot f(x, y'))
$$
令：
$$
g(x) = \log \sum_{y' \in \mathcal{Y}} \exp(v \cdot f(x, y'))
$$
上式得：
$$
\log p(y|x;v) = v\cdot f(x,y) -g(x)
$$
从上式可以看出，只要 $x$ 固定，那么 $\log p(y|x, v)$ 就是一个线性方程，所以模型的名称是对数线性模型。

好了，我们已经给出了对数线性模型的定义了，那这样一个模型和语言模型有什么关系呢？假设我们有下面一段话：

>   我是中国少年先锋队队员，我在队旗下宣誓，我决心遵守队章，在中国共产党和共青团的领导下，做个好队员。好好学习，好好工作，好好劳动，准备着为共产主义和祖国的伟大事业，贡献出一切（       ）！

我们首先令 $\pmb{h}$ 表示 n-gram，$y$ 表示需要填入括号中的词，计算 $v \cdot f(\pmb{h}, y)$ 内积，可以得到：

| $v \cdot f(\pmb{h}, y=力量)$ | $v \cdot f(\pmb{h}, y=小明)$ | $v \cdot f(\pmb{h}, y=什么)$ | $\cdots$ | $v \cdot f(\pmb{h}, y=这样)$ |
| :--------------------------: | :--------------------------: | :--------------------------: | :------: | :--------------------------: |
|            $8.1$             |            $0.5$             |            $-2.1$            | $\cdots$ |            $0.01$            |

要计算表格中的数据，首先要确定 $v$ 和 $f(\cdot)$，这个我们在后面的内容里详细介绍，现在先假定这两个是确定的，然后我们可以根据他们在给定 $\pmb{h}$ 和 $y$ 的情况下计算出相应的内积值。我们计算出来的内积可以是任意实数，包括正数、负数等等。

然后将计算出来的内积取 $e$ 指数：$\exp(v \cdot f(x,y))$，这样就保证了结果是非负数的了。

然后我们对结果求和：$\sum_{y' \in \mathcal{Y}} \exp(v \cdot f(x, y'))$，再利用求和结果对计算出的每一项进行归一化，这样我们就得到了一个对数线性模型。而该模型表示模型参数 $v$ 确定后，给定历史词 $\pmb{h}$ 的条件下，$y$ 出现的概率。

这样一个结果不就回到了 n-gram 语言模型 $p_{\theta}(w_j|\pmb{h})$，在确定参数 $\theta$ 的情况下，给定历史词 $\pmb{h}$ 的条件下，输出 $y$ 的概率吗？

# 3. 对数线性语言模型（Log-Linear Language Model）

为了和 n-gram 语言模型的符号保持一致，我们这里重写对数线性模型在语言模型上的定义：
$$
\begin{equation} \nonumber
\begin{aligned}
p_\theta(\pmb{X}=\pmb{x}) &= \prod_{j=1}^l p_{\theta}(W_j=w_j|W_{0:j-1}=w_{0:j-1}) \\\\
                          &= \prod_{j=1}^l \frac{\exp(\theta \cdot f(w_{0:j-1}, w_j))}{\sum_{w_j \in \mathcal{V}}\exp(\theta \cdot f(w_{0:j-1}, w_j))} \\\\
                          &= \prod_{j=1}^l \frac{\exp(\theta \cdot f(w_{j-n+1:j-1}, w_j))}{\sum_{w_j \in \mathcal{V}}\exp(\theta \cdot f(w_{j-n+1:j-1}, w_j))} (马尔科夫假设)\\\\
                          &= \prod_{j=1}^l \frac{\exp(\theta \cdot f(\pmb{h}, w_j))}{\sum_{w_j \in \mathcal{V}}\exp(\theta \cdot f(\pmb{h}, w_j))} \\\\
\end{aligned}
\end{equation}
$$
其中 $\theta$ 为模型参数，$f(\cdot)$ 表示特征投影。$\theta$ 和 $f(\cdot)$ 都是 $d$ 维向量。

## 3.1 特征

对于任意 $\{(x, y)|x\in \mathcal{X}, y\in \mathcal{Y}\}$，$f(x, y) \in \mathbb{R}^d$ 就是一个特征向量，其中 $f_k(x, y), k=[1,2,...,d]$，表示特征。每一维上的特征是可以任意定义的，只要是可以只通过 $\pmb{h}$ 和 $w_j$ 计算的。通常二元法进行定义，比如：

-   n-gram 特征。比如对于三元组特征 “$\langle 我,爱,北京 \rangle$”，如果 $\pmb{h} = \langle 我, 爱 \rangle，w_j=北京$，令 $f_k(\pmb{h}, w_j)=1$，否则 $f_k(\pmb{h},w_j)=0$；
-   空洞 n-gram 特征。比如 “$\langle 我,爱,北京 \rangle$”、“$\langle我,讨厌,北京\rangle$”、“$\langle 我,在,北京 \rangle$” 等等，只要 $\pmb{h} $ 的倒数第二个词是 “我”，并且 $w_j=北京$ ，那么就令 $f_k(\pmb{h}, w_j)=1$，否则 $f_k(\pmb{h}, w_j)=0$；
-   拼写特征，通常在英文语言模型上比较常见。比如判断一个词是否以大写字母开头，是否在词中包含数字，是否以元音字母开头等等，如果答案是 “是”，那么就令 $f_k(\pmb{h}, w_j)=1$，否则 $f_k(\pmb{h}, w_j)=0$；
-   类别特征，使用外部资源判断一个词是不是某种类别。比如是不是地名、是不是组织机构名、是不是名词、是不是形容词等，如果答案是 “是”，那么就令 $f_k(\pmb{h}, w_j)=1$，否则 $f_k(\pmb{h}, w_j)=0$；
-   $\cdots$

| n-gram 特征 | 空洞 n-gram特征 |                拼写特征                |              类别特征               | ...  | $\times \times$ 特征  |
| :---------: | :-------------: | :------------------------------------: | :---------------------------------: | ---- | :-------------------: |
|   $f_1=1$   |     $f_2=1$     | $f_3=0$<br />（$w_j=北京$ 不包含数字） | $f_3=1$ <br />（$w_j=北京$ 是地名） | ...  | $f_d(\pmb{h}, w_j)=0$ |

这样，我们对 “$\langle 我，爱，北京 \rangle$” 构建了一个 $f(x, y)=[1,1,0,1,...,0]$ 的特征向量。

总结一下我们构建某种特征的方式：
$$
f(\pmb{h}, w_j) = \begin{cases}
1 & 如果 \pmb{h}，w_j 满足某种条件 \\\\
0 & 否则
\end{cases}
$$
在特征构建的时候，$d$ 作为超参数出现，特征过多容易造成过拟合，特征不足会造成欠拟合。在构造特征的时候有一个重要的原则是必须遵守的，那就是区分性。也就是说任意两个不同的 $(\pmb{h}_i,w_i)$ 和 $(\pmb{h}_j, w_j)$ 都必须有不同的特征向量。所以通常 $d$ 的选取不应该小于 $V^{n}$ ，$n$ 表示 n-gram。

## 3.2 特征稀疏

上述方法构建特征向量有一个很严重的问题——特征稀疏化。通常自然语言处理任务用到的 $d$ 都会非常大。以 3-gram 为例，假设我们给每一个 $(w_{j-2}, w_{j-1}, w_j)$ 分配一个特征值（one-hot），那么 $d=V^3$。如此庞大的特征量，怎样使模型更有效率就成了一个问题。

对于任意给定的 $(\pmb{h},w_j)$ ，定义特征向量 $[f_1(\pmb{h}, w_j), f_2(\pmb{h}, w_j), ..., f_d(\pmb{h}, w_j)]$ 中 $f_k(\pmb{h}, w_j)=1$ 的个数：
$$
N(\pmb{h}, w_j) = \sum_{f(\pmb{h}, w_j)=1}f_k(\pmb{h}, w_j)
$$
通过实际观察我们会发现，通常 $N(\pmb{h}, w_j) \ll d$。最极端的情况，考虑 one-hot 特征向量，只有一个 1，其余全部是 0。

为了解决特征稀疏的问题，我们回过头去看对数线性语言模型的定义：
$$
p_\theta(\pmb{X}=\pmb{x}) = \prod_{j=1}^l \frac{\exp(\theta \cdot f(\pmb{h}, w_j))}{\sum_{w_j \in \mathcal{V}}\exp(\theta \cdot f(\pmb{h}, w_j))}
$$
通过观察我们会发现，上式的核心是 $\theta \cdot f(\pmb{h},w_j)$ 的计算：
$$
\theta \cdot f(\pmb{h}, w_j) = \sum_{k=1}^d \theta_k\cdot f_k(\pmb{h}, w_j)
$$
当 $f_k(\pmb{h},w_j)=0$ 时，$\theta_k \cdot f_k(\pmb{h}, w_j)=0$，所以内积最终的结果取决于 $f_k(\pmb{h}, w_j) \ne 0$ 的那些特征。

有了这样一个发现，我们就可以通过一些函数（比如哈希表）找到特征向量中的非零特征对应的索引：
$$
Z(\pmb{h}, w_j) = \{k: f_k(\pmb{h},w_j)=1\}
$$
有了 $Z(\pmb{h},w_j)$ 以后，可以得到：
$$
\sum_{k=1}^d \theta_k\cdot f_k(\pmb{h}, w_j) = \sum_{k\in Z(\pmb{h}, w_j)} \theta_k
$$
计算复杂度从 $\mathcal{O}(d)$ 降到 $\mathcal{O}(|Z(\pmb{h}, w_j)|)$。

## 3.3 Softmax

得到特征向量以后，我们可以通过 softmax 将特征向量的分数映射到概率分布上:
$$
\mathrm{softmax}([f_1, f_2, ..., f_d]) = \left[\frac{\exp(f_1)}{\sum_{k=1}^V\exp(f_k)}, \frac{\exp(f_2)}{\sum_{k=1}^V\exp(f_k)}, ..., \frac{\exp(f_d)}{\sum_{k=1}^V\exp(f_k)} \right]
$$
之所以使用 softmax 进行映射，是因为：

1.  保持了原来向量的单调性，即原来的特征 $f_a>f_b$，经过 softmax 映射以后仍然是 $f_a'>f_b'$;
2.  softmax 将原来的特征向量映射到了概率分布上，即向量所有元素和为 1；
3.  向量中的每个元素代表其对应的历史词 $\pmb{h}$ 出现的概率。

## 3.4 系数/权重

一旦我们将特征映射到特征向量以后，就可以计算 $\theta \cdot f(\pmb{h}, w_j)$ 了，其中 $\theta$ 我们称之为特征权重或者特征系数。我们可以将这样一个线性映射看成是 n-gram 的得分，特征权重 $\theta$ 中的每一维决定了特征向量中的每一个向量对 n-gram 最终得分的影响力。假设 $f_k(\pmb{h},w_j)=1$：

-   当 $\theta_k > 0$ 时，表明 $f_k(\pmb{h}, w_j)$ 在 n-gram 中起到了正向作用，即 $(\pmb{h},w_j)$ 出现的概率更高了；
-   当 $\theta_k<0$ 时，表明 $f_k(\pmb{h}, w_j)$ 在 n-gram 中起到了反向作用，即 $(\pmb{h},w_j)$ 出现的概率更低了；
-   当 $\theta_k=0$ 时，表明 $f_k(\pmb{h}, w_j)$ 在 n-gram 中没有任何影响，即 $(\pmb{h},w_j)$ 出现的概率不变；

知道了 $\theta$ 的意义，那么我们该怎么得到权重系数向量呢？还是靠验证集一个一个试吗？显然不靠谱，接下来就介绍一下 $\theta$ 的估计方法。

### 3.4.1 对数似然函数

之前我们定义了对数线性语言模型，每一个 n-gram 的对数概率是：
$$
\log p(w_j|\pmb{h},\theta)
$$
那么每个句子的对数概率为：
$$
L_\theta(\pmb{X}=\pmb{x}) = \sum_{j=1}^l \log p(w_j|\pmb{h},\theta)
$$
如果我们将 $\theta$ 作为变量，即：
$$
L_{\pmb{x}}(\theta) = \sum_{j=1}^l \log p(w_j|\pmb{h},\theta)
$$
可以将此式理解为：参数 $\theta$ 在多大程度上使语言模型接近训练数据。我们的目标是找到一套参数，使得语言模型最大程度接近训练数据：
$$
\theta = \arg \max_{\theta \in \mathbb{R}^d} L_{\pmb{x}}(\theta)
$$
在考虑如何对上式求解之前，我们思考这样一种情况：假设我们有一个三元组 $s = \langle 我，爱，北京\rangle$ 在训练集中只出现了一次，此时 $\theta$ 应该是什么样的。

根据概率最大原则，因为 $s$ 只出现一次，没有其他三元组和它分割概率，那么它出现的概率最大应该是 1，也就意味着：
$$
p_\theta(\pmb{X}=\pmb{我爱北京天安门}) = \prod_{j=1}^l \frac{\exp(\theta \cdot f(\pmb{h}, w_j))}{\sum_{w_j \in \mathcal{V}}\exp(\theta \cdot f(\pmb{h}, w_j))}=1
$$
上式分子为（假设特征向量只有$f_k=1$，其余全部为零）：
$$
\exp(\theta_k)
$$
分母为：
$$
\sum_{w_j \in \mathcal{V}}\exp(\theta_k)
$$
当且仅当 $\theta_k \rightarrow \infty$ 时
$$
\lim_{\theta_k \rightarrow \infty} \frac{\exp(\theta_k)}{\sum \exp(\theta_k)} = 1
$$
也就是说，这种情况会导致特征权重接近无穷，这显然是不对的。为了解决这个问题，通常采用的方法是加入正则化。

### 3.4.2 正则化

正则化的方法有很多，比如 L1 正则化、L2 正则化等等。常用的正则化是 L2 正则化， 定义如下：
$$
||\theta||^2 = \sum_k \theta_k^2
$$
其中 $||\theta||$ 表示向量 $\theta$ 的欧拉长度，即 $||\theta|| = \sqrt{\sum_k \theta_k^2}$。经过正则化修正的目标函数为：
$$
L'(\theta)=\sum_{j=1}^l \log p(w_j|\pmb{h};\theta) - \frac{\lambda}{2} \sum_k\theta_k^2
$$
其中 $\lambda>0$ 是一个超参数。

使用正则化避免出现参数过大的原理是这样的：上式第一项的意义是参数 $\theta$ 能使语言模型在多大程度上接近训练数据；第二项的意义是对参数过大的惩罚项：它希望参数尽可能接近零，因为欧拉距离是大于等于零的，$\lambda$ 也是一个大于零的数，也就是说第二项是一个大于零的数，要想使得 $L'(\theta)$ 尽可能大，就需要上式中第一项尽可能大，而第二项尽可能小。

### 3.4.3 梯度下降

有了目标函数以后，我们可以通过梯度下降算法对参数进行更新，直到参数收敛。

参数更新过程如下：

>   -   初始化 $\theta=\pmb{0}$ （元素全部是 0 的向量）
>   -   重复下面的过程：
>       1.  计算 $\delta_k=\frac{dL'(\theta)}{d\theta}, k=[1,2,...,d]$;
>       2.  计算 $\beta^*=\arg \max_{\beta \in \mathbb{R}}L'(\theta+\beta \delta)$，
>       3.  更新 $\theta \leftarrow \theta + \beta^*\delta$

其中
$$
\frac{dL'(\theta)}{d_{\theta_k}} = \sum_{i=1}^n f_k(\pmb{h}^{(i)}, w_j^{(i)})-\sum_{i=1}^n\sum_{w_j\in\mathcal{Y}} p(w_j|\pmb{h}^{(i)};\theta)f_k(\pmb{h}_{(i)}, w_j)-\lambda\theta_k
$$

 # 4. 小结

对数线性模型的出现为我们打开了一条通往神经网络语言模型的道路，我们发现对数线性语言模型中蕴含的思想已经具备了神经网络语言模型的雏形。尤其是将 n-gram 扩展到了特征这一概念，为后来的特征向量分布式表示奠定了基础。这里使用的特征构建方法虽然简单，但是与分布式特征向量表示相比也存在着优点，那就是每一维上的特征都具有具体的含义，这就意味着，该模型是具有可解释性的。神经网络语言模型对特征的自动构建也有很多对数线性语言模型特征向量不具备的优势，比如无需人工参与、可以大幅度缩减参数维度、可计算词与词之间的语义关系等。总而言之，对数线性语言模型简约而不简单。