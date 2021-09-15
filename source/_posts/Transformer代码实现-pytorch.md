---
type: blog
title: Transformer代码实现-Pytorch版
top: false
cover: true
toc: true
mathjax: true
date: 2019-09-11 16:35:30
password:
summary: Transformer代码实现
tags: [Transformer, pytorch]
categories: [NLP]
body: [article, comments]
gitalk:
  id: /wiki/material-x/
---

前面介绍了Transformer的模型结构，最后也给出了`pytorch`版本的代码实现，但是始终觉得不够过瘾，有些话还没说清楚，因此，这篇文章专门用来讨论Transformer的代码细节。

<!-- more -->

本文主要参考了：[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。这篇文章是哈佛大学OpenNMT团队的工作，所以在正式进入话题之前要先把代码环境搭建好。`Pytorch`的安装网上有详细的教程这里不再赘述，只简单提一点，直接下载安装的话可能会速度比较慢，甚至下载失败，可以使用国内清华大学的镜像进行安装：

- 添加清华大学镜像：

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

- 添加`pytorch`镜像：

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

# 1. 前期准备

导入相应的包

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
```

# 2. 模型结构

大多数有竞争力的神经序列转换模型都有*encoder-decoder*结构，其中*encoder*部分将输入序列$(x_1, x_2, ..., x_n)$映射到一个连续表示的序列$\mathbf{z}=(z_1, z_2, ..., z_n)$中。给定$\mathbf{z}$，*decoder*再生成一个输出序列$(y_1, y_2, ..., y_m)$。

```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

```python
class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

![](https://img.vim-cn.com/3a/78ea12dca1ce0f99f9a9705466afc16c58c3cf.png)

*encoder*和*decoder*都是由6个这样的结构堆叠而成的：

```python
def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

## 2.1 Encoder

*encoder*是由6个相同的模块堆叠在一起的：

```python
class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

每个*encoder block*由*Multi-head Attention*和*Feed Forward*两个sub-layer组成，每个sub-layer后面会接一个layer normalization：

```python
class LayerNorm(nn.Module):
    """
    Construct a layer normalization module.
    See https://arxiv.org/abs/1607.06450 for detail
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mwan(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

sub-layer和layer normalization之间使用残差方式进行连接（进行残差连接之前都会先进行Dropout）：

```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    See http://jmlr.org/papers/v15/srivastava14a.html for dropout detail
    and https://arxiv.org/abs/1512.03385 for residual connection detail.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))
```

模型中，为了使`x + self.dropout(sublayer(self.norm(x)))`能够正常运行，必须保证`x`和`dropout`的维度保持一致，论文中使用的$d_{model}=512$（包括embedding层）。

```python
class EncoderLayer(nn.module):
    """
    Encoder block consist of two sub-layers (define below): 
    - multi-head attention (self-attention) 
    - feed forward.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        
    def forward(self, x, mask):
        """
        Encoder block.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

## 2.2 Decoder

*Decoder*同样是由6个相同的模块堆叠在一起的：

```python
class Decoder(nn.module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

与*encoder block*不同的是，在*decoder block*的*Multi-head attention*和*Feed forward*之间还会插入一个*Multi-head attention*，这个*attention*中的key和value来源于*encoder*的输出。

```python
class DecoderLayer(nn.module):
    """
    Encoder block consist of three sub-layers (define below): 
    - multi-head attention (self-attention) 
    - encoder multi-head attention 
    - feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn =  src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Decoder block.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

为了保证解码过程中第$i$个位置的输出只依赖于前面已有的输出结果，在*decoder*中加入了**Masking**:

```python
def subsequent_mask(size):
    """
    Mask out subsequent position.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

下图的attention mask展示了每个目标词（行）可以看到的位置（列），黄色表示可以看到，紫色表示看不到。

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_31_0.png)

# 3. 模型细节

上面我们实现了模型的整体结构，下面我们来实现其中的细节。前面我们提到，每个*encoder block*有两个sub-layer：*Multi-head attention*和*feed forward*，虽然*decoder block*有三个sub-layer，但是两个都是*Multi-head attention*，说到底还是只有*Multi-head attention*和*feed forward*。

## 3.1 Multi-Head Attention

之前我们介绍的时候讲到所谓*Multi-Head Attention*是有两部分组成：*Multi-Head*和*Attention*。

结构如下：

![](https://img.vim-cn.com/b1/e4bc841abc55d366813340f92f6696c5d59e95.png)
$$
\mathrm{MultiHead}(Q, K, V) = Concat(head_1, ..., head_h)\mathbf{W}^O
$$
其中$head_i$就是Attention，即$head_i=\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$具体结构如下图：

![](https://img.vim-cn.com/ed/97e04d7d6067cb360e8fef1d29cf41978d353e.png)

先来看下*Scaled Dot-Product Attention*得出具体实现：

```python
def attention(query, key, value, mask=None, dropout=None):
        """
        Compute `Scale Dot-Product Attention`.
        
        :params query: linear projected query maxtrix, Q in above figure right
        :params key: linear projected key maxtrix, k in above figure right
        :params value: linear projected value maxtrix, v in above figure right
        :params mask: sub-sequence mask
        :params dropout: rate of dropout
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.mask_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
```

下面我们就可以实现*Multi-head Attention*了：

```python
class MultiHeadAttention(nn.Module):
    """
    Build Multi-Head Attention sub-layer.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        :params h: int, number of heads
        :params d_model: model size
        :params dropout: rate of dropout
        """
        super(MultiHeadAtention, self).__init__()
        assert d_model % h == 0
        # According to the paper, d_v always equals to d_k
        # and d_v = d_k = d_model / h = 64
        self.d_k = d_model // h
        self.h = h
        # following K, Q, V and `Concat`, so we need 4 linears
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Implement Multi-Head Attention.
        
        :params query: query embedding matrix, Q in above figure left
        :params key: key embedding matrix, K in above figure left
        :params value value embedding matrix, V in above figure left
        :params mask: sub-sequence mask
        """
        if mask is not None:
            # same mask applied to all heads
            mask = mask.unsequeeze(1)
        n_batch = query.size(0)
        # 1. Do all the linear projections in batch from d_model to h x d_k
        query, key, value = [l(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 2. Apply attention on all the projected vectors in batch
        x, self.attn = self.attention(query, key, value, mask=mask)
        # 3. `Concat` using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

*Transformer*中*Multi-Head Attention*有三种用法：

1. 在*decoder*层中，中间的*Multi-Head Attention*模块中query来源于前置*Masked Multi-Head Attention*模块，而key和value来源于*encoder*层的输出， 这一部分模仿了典型的*seq2seq*模型中的*encoder-decoder*注意力机制；
2. 在*encoder*层中，所有的query, key, value都来源于前一个*encoder*层的输出；
3. 类似的在*decoder*层中，有一个*Masked Multi-Head Attention*模块，其中*Masked*是因为在进行解码的过程中，我们是从左向右一步一步的进行解码，对于模型来说右侧的信息是缺失的，因此不应该对左侧的信息产生干扰，因此在模型中我们令相应位置的值为$\infty$。

## 3.2 Position-wise Feed-Forward Networks

*Feed forward*部分是由两个*Relu*线性变换组成的，在同一个*block*内的的不同位置使用相同的参数，但是不同*block*使用不同的参数。
$$
\mathrm{FFN(x)} = \max(0, xW_1 + b_1)W_2 + b_2
$$
这个操作类似于卷积核大小为1的卷积操作。              

```python
class PositionwiseFeedForward(nn.module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, f_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
```

## 3.3 Embedding和Softmax

*Transformer*中使用预训练的word embeddings，并且输入和输出的word embedding保持一致。和其他模型不同的是word embedding并不是直接进入模型，而是乘上一个缩放因子$\sqrt{d_{model}}$：

```python
class Embeddings(nn.module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab, d_model)
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

## 3.4 Position Encoding

由于单纯的注意力机制没有有效的利用序列的顺序信息，因此作者在*Transformer*中加入了位置编码，用来抓住序列中的位置信息。
$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

其中$pos$指得是位置索引，$i$是第$pos$个位置上对应向量的第$i$维。对序列的位置进行编码后，将输入序列和位置编码进行相加，得到一个新的输入序列。

```python
class PositionEncoding(nn.module):
    """
    Implements Position Encoding.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the position encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
            return self.dropout(x)
```

至此， 整个模型各个模块我们已经搭建好了，最后进行总装。

# 4. Full Model

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Construct Transformer model.
    
    :params src_vocab: source language vocabulary
    :params tgt_vocab: target language vocabulary
    :params N: number of encoder or decoder stacks
    :params d_model: dimension of model input and output
    :params d_ff: dimension of feed forward layer
    :params h: number of attention head
    :params dropout: rate of dropout
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    position = PositionEncoding(d_model, dropout)
    model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout),N),   
                Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                Generator(d_model, tgt_vocab)
    )
    
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```

至此Transformer模型已经完成了，下面介绍是整个模型的训练过程以及机器翻译过程的一些技巧和常用工具的介绍，没有兴趣的话到这里就可以结束了。

# 5. 模型训练

本节快速介绍一些在训练*encoder-decoder*模型过程中常用的工具，首先定义一个*batch*对象用来获取训练所需的源句子和目标句子，以及构建*masking*。

## 5.1 Batches and Masking

```python
class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsequeeze(-2)
        if tgt is not None:
            self.tgt = tgt[;, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgtg_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
            
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsequeeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        )
        return tgt_mask
```

接下来我们创建一个通用的训练和计算得分的函数用于跟踪损失。我们传入一个通用的用于更新权重的损失函数。

## 5.2 Training Loop

```python
def run_epoch(data_iter, model, loss_compute):
    """
    Standard training and logging function.
    """
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src,
                            batch.tgt,
                            batch.src_mask,
                            batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start_time
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" 
                  % (i, loss / batch.ntokens, tokens / elapsed))
            start_time = time.time()
            token = 0
        return total_loss / total_tokens
```

## 5.3 Training Data and Batching

论文使用的数据集：

- WMT 2014 English-German dataset： 4.5 million sentence pairs
- WMT 2014 English-French dataset： 36 M sentence pairs

由于*Transformer*本身训练需要的资源较多，而且上面的数据集过多，本文只是从原理上实现算法，不需要训练一个实际可用的模型，因此并没有在这两个数据集上进行训练。

```python
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """
    Keeping augumenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```

## 5.4 Optimizer

论文中使用`Adam`优化器，其中$\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$；学习率根据公式$l_{rate} = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$来确定，该式意味着在开始的$warmup\_steps$循环内学习率是线性增加的，达到一定程度后学习率开始下降， 论文中的$warmup\_step=4000$，这是一个超参数。

```python
class NoamOpt:
    """
    Optimizer warp that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optmizer.step()
        
    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step
        return self.factor * self.model_size ** (-0.5) * \
               min(step ** (-0.5), step * self.warmup ** (-1.5))
    
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0],d_model, 2, 4000, 
                       torch.optim.Adam(model.parameters(), 
                                        lr=0, betas=(0.9, 0.98), eps=1e-9))
```

下面给出了三组不同的优化器超参数的例子，直观的感受学习率的变化。

```python
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000, [[opt.rate(i) for opt in opts] for i in range(1, 20000)]))
plt.legend(["512:4000", "512:8000", "256:4000"])
```

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_69_0.png)

## 5.5 正则化

训练过程中作者使用了*label smoothing* $\epsilon_{ls}=0.1$， 虽然这样对*Perplexity*有所损伤， 但是提高了整体的BLEU值。这里我们用KL散度损失实现了*label smoothing*，并且使用分布式目标词分布用以替代*one-hot*分布。

```python
class LabelSmoothing(nn.Module):
    """
    Implements label smoothing.
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsequeeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask_dim() > 0:
            true_dist.index_fill(0, mask.sequeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
```

下面给出一个例子说明基于置信度的词分布看起来是什么样的

```python
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([
    [0, 0.2, 0.7, 0.1, 0],
    [0, 0.2, 0.7, 0.1, 0],
    [0, 0.2, 0.7, 0.1, 0]
])
v = crit(Variable(predict.log()),
         Variable(torch.LongTensor([2, 1, 0])))
plt.imshow(crit.true_dist)
```

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_74_0.png)

如果模型对一个给定的选择给出非常大的置信度，*Label smoothing*就会开始对模型进行惩罚。

```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([
        [0, x/d, 1/d, 1/d, 1/d]
    ])
    return crit(Variable(predict.log()),
                Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
```

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_76_0.png)

# 6. 第一个例子

在正式在真实的数据上做实验之前，我们可以先在一个随机生成的数据集上实验，目标是生成和源序列相同的序列，例如源序列是“I have a dream”，我们的目标是将序列输入到模型，然后输出这个序列。

## 6.1 合成数据

```python
def data_gen(V, batch, nbatches):
    """
    Generate random data for src-tgt copy task.
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
```

## 6.2 损失计算

```python
class SimleLossCompute:
    """
    A simple loss compute and train function.
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
```

## 6.3 Greedy Decoding

```python
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, 
                                     betas=(0.9, 0.98), eps=1e-9))
for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
             SimpleLossCompute(model.generator, criterion, model_opt))
    model.evel()
    print(run_epoch(data_gen(V, 30, 5), model, 
                   SimpleLossCompute(model.generator, criterion, None)))
```

实际的翻译模型一般使用*beam search*，这里为例简化代码，我们使用贪婪编码。

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                          Variable(ys), 
                          Variable(subsequent_mask(ys.size(1))
                                   .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                       torch.ones(1, 1).type_as(src.data).fill_(next_word)],
                       dim=1)
    return ys
    
model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
```

# 7. A Real World Example

这里我们使用IWSLT German-English Translation 数据集，这个数据集比论文中使用的数据集小得多，但是能够检验整个模型。下面我们还会演示怎样在多GPU上进行训练。

```shell
pip install torchtext spacy
python -m spacy download en
python -m spacy download de
```

## 7.1 数据加载

```python
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    
    def tokenize_de(text):
        return [tok.text for tok in spacy.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for  tok in spcy.tokenizer(text)]
    
    BOS_WORD = '<S>'
    EOS_WORD = '</S>'
    BLANK_WORD = '<blank>'
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                    eos_token=EOS_WORD, pad_token=BLANK_WORD)
    
    MAX_LEN = 100
    TRAIN, VAL, TEST = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
            len(vars(x)['trg']) <= MAX_LEN
    )
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq = MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq = MIN_FREQ)
```

## 7.2 Iterators

```python
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p  in data.batch(d, self.batch_size * 100):
                    p_batch =  data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b
             self.batches = pool(self.data(), self.random_shuffler)
            
         else:
             self.batches = []
                for b in data.batch(self.data(), self.batch_size,
                                   self.batch_size_fn):
                    self.batches.append(sorted(b, key=self.sort_key))
                    
    def rebatch(pad_idx, batch):
        """
        Fix order in torchtext to match ours.
        """
        src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        return Batch(src, trg, pad_idx)
```

## 7.3 Multi-GPU Training

最后，为了加速训练，我们使用多GPU进行训练。方法就是在训练过程中将生成词的过程分成多份在多个GPU上并行处理。

我们使用pytorch的原生库来实现：

- `replicate` - 将模块分割放进不同的GPU上；
- `scatter` - 将不同的batch放进不同的GPU上；
- `parallel_apply` - 将不同的batch放到对应的GPU中的模块中；
- `gather` - 将分散的数据重新集合到同一个GU上；
- `nn.DataParallel` - 一个特殊的模块集合，用来在评估模型之前调度上面那些模块

```python
# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
```

下面我们就可以构造模型了：

```python
# GPUs to use
devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
```

## 7.4 训练模型

```python
if False:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")
```

模型一旦训练好了我们就可以用来翻译了。这里我们以验证集的第一个句子为例，进行翻译：

```python
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
```

# 8. 注意力可视化

尽管贪婪编码的翻译看起来不错，但是我们还想看看每层注意力到底发生了什么：

```python
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
```

> Encoder Layer 2

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_1.png)

> Encoder Layer 4

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_3.png)

> Encoder Layer 6

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_5.png)

> Decoder Self Layer 2

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_7.png)

> Decoder Src Layer 2

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_9.png)

> Decoder Self Layer 4

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_11.png)

> Decoder Src Layer 4

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_13.png)

> Decoder Self Layer 6

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_15.png)

> Decoder Src Layer 6

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_119_17.png)

# 9. 参考资料

[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

