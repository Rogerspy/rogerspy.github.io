---
type: article
title: Compact word vectors with Bloom embeddings
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2022-05-17 10:33:56
password:
summary:
tags: 词向量压缩
categories: 博客转载
---

<img src="https://explosion.ai/static/c570bfb56770b9ce58baf9de2b257d8e/338d3/bloom-embeddings.png" style="zoom: 33%;" />

通常 word embedding 表占用的内存是非常大的，100 万个 32-bit 的浮点数内存是 4MB，那么 100 万个300 维的向量至少要占用 1.2 GB 的内存。这种内存消耗会严重限制 word embeeding 的应用场景。

<!--more-->

通常有三种方式降低 word embedding 的内存占用：

- 减小词表大小
- 降低词向量维度
- 降低每个维度浮点数的精度

以上三种方法是非常有效的，但是也会带来一些问题。比如减小词表大小会更容易带来未登录词的问题，降低词向量维度会降低词向量表达能力，降低精度通常也会影响模型效果。

吹了以上三种方法之外，还有一种方式：

- cheat，即使用一种概率数据结构。概率数据结构是一个神经网络模型，可以被广泛使用。但是神经网络模型的问题就是不够直观（黑盒子），所以我们起名为“cheat”。下面我们详细介绍一下。

# 1. Bloom embeddings 算法

常规的 word embedding 矩阵是将每个词映射到唯一的 ID 上，通常这些 ID 是序列化的，比如词表中有 100 个词，那么这些词的 ID 就是 `range(100)`。然后将这些 ID 与矩阵中的每一行对应起来，这样就相当于将每一行与词对应起来了。比如 “apple” 的 ID 是 5，那么它的词向量就对应 embedding 矩阵中的第 5 行。这种方式对 embedding 矩阵的行数是不限制的，即词表有多大那么矩阵就有多少行。

Bloom embedding 是可以限制 embedding 矩阵的行数的（但是不限制词表的大小），当词表大小超过了 embedding 矩阵行数的时候就会出现多个词共享词向量的情况。一个解决方式就是设置一个特殊的向量，0-98 的词分别对应相应的向量，其他 ID 的词就全部对应到第 99 行的向量上。

显然这种解决方案太过暴力，这就相当于将太多的词都设置成了未登录词（<unknown>）。但是如果我们不止一行这种特殊向量呢？比如我们设置 10 行特殊向量，0-89 编号的词分别对应相应的向量，其他词随机在这 10 行特殊向量中选择一个向量呢？这就相当于我们有 10 个不同的未登录词 <unknow0>...<unknow9>。

```python
def get_row(word_id, number_vector=100, number_oov=10):
    number_known = number_vector - number_oov
    if word_id < number_known:
        return word_id
    else:
        return number_known + (word_id % number_oov)
```

如果我们将所有的未登录词都映射到同一个向量上去，对于模型来说就无法区分不同的训练数据。比如，有两个句子，其中每句话中有一个词能够很强烈的暗示这个句子的情感取向，其中一个是积极情绪，另一个是负面情绪，恰恰这两个词都是在特殊向量上，那么模型就无法区分这两个句子的区别了。而如果我们有多个特殊向量，而我们又足够幸运，这两个词分别对应到了不同的特殊向量上，那么模型就可以得到两个不同的结果。

如果上面这种假设是成立的，那么我们为什么不多设置一些特殊向量呢？Bloom embedding 就是一个极端的模型，将词表中的每一个词都设置成未登录词，然后 embedding 矩阵中的每一个向量都是特殊向量。

![](https://explosion.ai/1ba0ccfd147a86fe5811a8273f1e735a/bloom-embeddings_normal_vs_hashed.svg)



这种极端方法看起来很奇怪，而且还不一定好用。其实真正让 Bloom embedding 算法发挥作用的是下一步：通过简单重复多次相同的操作，我们可以极大提升词的分辨率，并且有比向量更多的唯一词向量表示：

```python
import numpy as np
import mmh3

def allocate(n_vectors, n_dimensions):
    table = np.zeros((n_vectors, n_dimensions), dtype='f')
    table += np.random.uniform(-0.1, 0.1, table.size).reshape(table.size)
    return table

def get_vector(table, word):
    hash1 = mmh3.hash(word, seed=0)
    hash2 = mmh3.hash(word, seed=1)
    row1 = hash1 % table.shape[0]
    row2 = hash2 % table.shape[0]
    return table[row1] + table[row2]

def update_vector(table, word, d_vector):
    hash1 = mmh3.hash(word, seed=0)
    hash2 = mmh3.hash(word, seed=1)
    row1 = hash1 % table.shape[0]
    row2 = hash2 % table.shape[0]
    table[row1] -= 0.001 * d_vector
    table[row2] -= 0.001 * d_vector
    return table
```

在上面的例子中，我们使用哈希函数得到两个 key，这两个 key 不太可能像词一样发生碰撞，然后将两个向量相加就可以得到更多词的唯一表示了。

**例子**

下面我们来看一个小例子：

```python
vocab = ['apple', 'strawberry', 'orange', 'juice',
         'drink', 'smoothie', 'eat', 'fruit',
         'health', 'wellness', 'steak', 'fries',
         'ketchup', 'burger', 'chips', 'lobster',
         'caviar', 'service', 'waiter', 'chef']
```

我们将这些词映射到 2-d 向量中。正常情况下，我们会得到一个 `(20, 2)` 的矩阵。我们使用上面的方法生成一个`(15, 2)` 的矩阵：

```python
normal = np.random.uniform(-0.1, 0.1, (20, 2))
hashed = np.random.uniform(-0.1, 0.1, (15, 2))
```

- 常规矩阵

  在常规矩阵中，我们将每个词映射进矩阵：

  ```python
  word2id = {}
  def get_normal_vector(word, table):
      if word not in word2id:
          word2id[word] = len(word2id)
      return normal[word2id[word]]
  ```

- hashed 矩阵

  哈希矩阵只有 15 行，所以有一些词必须共享向量。我们使用哈希函数生成哈希值，然后将哈希值修正到 `range(0,15)` 范围内。需要注意的是，我们要计算每个 key 的多个不同的哈希值，所以 python 内置的哈希函数不太方便，我们使用 `MurmurHash`：

  ```python
  hashed = [mmh3.hash(w,1) % 15 for w in vocab]
  assert hashes1 == [3, 6, 4, 13, 8, 3, 13, 1, 9, 12, 11, 4, 2, 13, 5, 10, 0, 2, 10, 13]
  ```

  如你所见，一些 key 被多个词共享，然而还有 2/15 的 key 是闲置的。显然这种情况并不理想。 比如词表中的“strawberry” 和 “heart” 就没有了区分性，模型无法得知我们用的是那个词。

  为了解决这个问题，我们再次对每个词使用一次哈希函数，这一次我们用不同的 seed：

  ```python
  from collections import Counter
  
  hashes2 = [mmh3.hash(w, 2) % 15 for w in vocab]
  assert len(Counter(hashes2).most_common()) == 12
  ```

  这次更糟，有 3 个闲置 key， 但是我们会发现：

  ```python
  assert len(Counter(zip(hashes1, hashes2))) == 20
  ```

  通过两次哈希结果的组合，我们就可以得到 20 个唯一的组合。这就意味着我们将两个向量相加就可以得到唯一的向量表示了：

  ```python
  for word in vocab:
      key1 = mmh3.hash(word, 0) % 15
      key2 = mmh3.hash(word, 1) % 15
      vector = hashed[key1] + hashed[key2]
      print(word, '%.3f %.3f' % tuple(vector))
  ```

  ```
  apple 0.161 0.163
  strawberry 0.128 -0.024
  orange 0.157 -0.047
  juice -0.017 -0.023
  drink 0.097 -0.124
  smoothie 0.085 0.024
  eat 0.000 -0.105
  fruit -0.060 -0.053
  health 0.166 0.103
  wellness 0.011 0.065
  steak 0.155 -0.039
  fries 0.157 -0.106
  ketchup 0.076 0.127
  burger 0.045 -0.084
  chips 0.082 -0.037
  lobster 0.138 0.067
  caviar 0.066 0.098
  service -0.017 -0.023
  waiter 0.039 0.001
  chef -0.016 0.030
  ```

这样我们就有了一个函数，可以将 20 个词 映射到 20 个唯一的向量上去，但是我们的词向量却只有 15 行，节省了 1/4 的内存。现在的问题是，我们虽然将词映射到了向量上去，但是这种映射是否有效？

我们现在做一个小实验，先随机设置一个常规 word embedding 矩阵当作每个词的真实词向量，然后我们看下是否能够通过压缩过的 embedding 矩阵得到与之匹配的向量：

```python
import numpy as np
import mmh3

np.random.seed(0)

nb_epoch = 20
learn_rate = 0.001
nr_hash_vector = 15

words = [str(i) for i in range(20)]
true_vectors = np.random.uniform(-0.1, 0.1, (len(words), 2))  # 真实的词向量矩阵
hash_vectors = np.random.uniform(-0.1, 0.1, (nr_hash_vector, 2))  # 压缩后的词向量矩阵
examples = list(zip(words, true_vectors))

for epoch in range(nb_epoch):
    np.random.shuffle(examples)
    loss = 0.0
    for word, truth in examples:
        key1 = mmh3.hash(word, 0) % nr_hash_vector
        key2 = mmh3.hash(word, 1) % nr_hash_vector
        
        hash_vector = hash_vectors[key1] + hash_vectors[key2]
        
        diff = hash_vector - truth
        
        # 模拟梯度下降
        hash_vectors[key1] -= learn_rate * diff
        hash_vectors[key2] -= learn_rate * diff
        loss += (diff*2).sum()
    print(epoch, loss)
```

我们可以得到：

```
0 -4.552187444900751
1 -4.513732579131083
2 -4.47518179590089
3 -4.439109966571761
4 -4.4016736165408386
5 -4.3649578351872
6 -4.330003220666162
7 -4.294276613407629
8 -4.260200584655477
9 -4.22524699343236
10 -4.190274216061722
11 -4.154823580829194
12 -4.119579998535728
13 -4.087484288266794
14 -4.053525036916957
15 -4.020518293389426
16 -3.987864913164988
17 -3.956116312323212
18 -3.9218190519236003
19 -3.8946058240614283
```

实际上，我们测试时将 `epoch` 设置成 2000 时，loss 仍然在下降。所以我们可以得出结论：经过哈希压缩的 embedding 矩阵仍然是有效的。

进一步地，我们可以思考下面几个问题：

- 当 `nr_hash_vector` 发生变化时，`loss` 怎么变？
- 如果移除 `key2`，`loss` 会变大吗？
- 如果增加 key 的个数会发生什么？
- 如果词表变大会发生什么？
- 如果词向量的维度增加会发生什么？
- 哈希 embedding 对初始化条件有多敏感？如果我们改变随机种子对结果有多大影响？

# 2. HashedEmbed

`HashedEmbed` 是用 4 个小哈希表，分别对应词的四种特征，利用 4 个 key 分别从每个表中抽取一个向量，将这四个向量相加得到最后的词向量。四种特征分别是 `ORTH` ， `PREFIX`，`SUFFIX`，`SHAPE`。比如我们用 5000 行的矩阵表示 `ORTH`，2500 行的矩阵表示 `PREFIX`，2500 行的矩阵表示 `SUFFIX` 以及 2500 行的矩阵表示 `SHAPE`。最终整个 embedding 矩阵只有12500 行 96 维，只需要占用 5 MB的存储空间。下面是 `Apple` 的例子：	

![](https://explosion.ai/399b8d35f03cf1a4eb5dfefe58645255/bloom-embeddings_tok2vec.svg)

从模型尺寸的角度来考虑，这种方式还有另一个好处。由于给定相同的哈希函数的话，相同的字符串就是映射到相同的哈希值，因此，我们不需要再维护一个词表。任意字符串都可以通过哈希函数映射到对应的词向量上，而不需要词表的 ID 来映射。

# 3. floret: Bloom embeddings for fastText

`floret` 是一个用 Bloom embedding 进行改写的 fasttext 模型变种。Fasttext 使用词的 n-gram 特征作为输入，一个词向量是通过对该词的所有 ngram 特征加和求平均得到的。比如：

```
<apple>
<app
appl
pple
ple>
```

fasttext 还支持可变 n-gram，比如 `n=[4,5,6]`：

```
<apple>
<app
appl
pple
ple>
<appl
apple
pple>
<apple
apple>
```

通过使用 subword 的方法，fasttext 可以得到更有用的输入向量，比如 `appletrees` 在词表中没有，如果不用 subword 的话，那就会映射到 `unknown` 上去，而是用 subword 的话，词表中是存在 `apple` 和 `tree` 的，这样就能得到更好的表达。

fasttext 使用两个独立的表分别存储 word 和 subword：

![](https://explosion.ai/b5ba6d2c139cba49ba2be1a9b9b0bd8f/bloom-embeddings_separate_words_subwords.svg)

两个表的典型大小都是 2M 行，假设词向量的维度是 300 维，那么两个表需要占用至少 4GB 的存储空间！

这个时候就需要 bloom embedding 出场了：

- 将 word 和 subword 存储在同一个表中（典型尺寸为 50-200 K）：

  ![](https://explosion.ai/c7527d7f3989176c781d35811797a339/bloom-embeddings_combined_words_subwords.svg)

- 将每一个 token 映射到多行：

  ![](https://explosion.ai/0ae08f6a499fd4569bb96f6177c503d2/bloom-embeddings_floret_bloom.svg)

这样，我们可以将存储降到 100 MB 以下！

下图是实验结果：

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs20220517105210.png)

# Reference

[Compact word vectors with Bloom embeddings](https://explosion.ai/blog/bloom-embeddings?continueFlag=b6ca8b587fbbab42bb48a07f3703ff16)





