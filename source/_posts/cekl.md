---
type: article
title: 交叉熵与 KL 散度
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-03-08 21:52:17
password:
summary:
tags: cross-entropy
categories: 笔记
---

![](https://images.squarespace-cdn.com/content/v1/54e50c15e4b058fc6806d068/1494401025139-ODE7CP2043TS1CO9MQSN/biting-worms.jpg?format=500w)

<!--more-->

首先假设一个场景：假设我们是一个空间科学家，现在正在造访一个遥远的行星，在这颗行星上发现了一种蠕虫。我们发现这些蠕虫的一半有10颗牙齿，但是因为某些原因很多蠕虫的牙齿是有缺失的。我们收集了很多蠕虫统计了他们的牙齿数量

<img src="https://images.squarespace-cdn.com/content/v1/54e50c15e4b058fc6806d068/1494396254926-SWY85XI22T1M4Q1ZEPCO/empirical-distribution-of-data.png?format=750w" style="zoom:50%;" />

我们想把这些数据发送回地球，有一个问题就是距离太远，把这么多的数据发送回去代价太高，我们想把数据压缩到一个只需要几个参数的模型上去。

第一个模型就是均匀分布：

<img src="https://images.squarespace-cdn.com/content/v1/54e50c15e4b058fc6806d068/1494397124633-IC3E9LB2IML2JXHJQVFQ/uniform-approximation.png?format=750w" style="zoom:50%;" />

很显然，跟我们的实际的牙齿概率分布区别还蛮大的，所以我们想到另一个分布——二项分布：

<img src="https://images.squarespace-cdn.com/content/v1/54e50c15e4b058fc6806d068/1494397201106-RKMWRQ4GUNY1ZTKCM1S0/binomial-approximation.png?format=750w" style="zoom:50%;" />

为了对比这两种分布哪个更符合实际，我们把这些数据画到同一张图上去：

<img src="https://images.squarespace-cdn.com/content/v1/54e50c15e4b058fc6806d068/1494397358518-09MZORGNU1VQK4EBQ1ZL/all-approximations.png?format=750w" style="zoom:50%;" />

这样我们还是没办法直观的看出哪个分布更好一些。我们最初的设想是以最小的信息损失将我们的发现传回地球。这两个模型都可以以非常少的参数完成任务，但是哪一个模型的信息损失更小呢？要探究这个问题，首先我们要知道什么是信息？

# 信息量

我们那需要做的一个基础工作就是量化信息。我们先对信息量做一个定义：

>   所谓信息量就是我们对一个事件进行编码所需要的比特数。

直观上来说，越是罕见的事我们要对其进行编码，就需要更多的信息来对它进行编码。也就是说，概率越低的事件，信息量越高。举个例子：

>   事件 $I$：人都会死。
>
>   事件 $II$：我可以开游艇，住别墅。

对于事件 $I$ 来说，我们通常会吐槽 “你说点有用的话！”就是因为这是一件人人都知道，没有任何信息的话。但是如果有人告诉你事件 $II$，这句话透露出来的信息是，这个人很有钱，他过着锦衣玉食的生活。说明事件 $II$ 的信息量比事件 $I$ 的信息量更大。对于事件 $I$ 来说，是必然会发生的，也就是概率为 $1$，而事件 $II$ 却是小概率事件。这样我们可以得到一个结论：

-   小概率事件：信息量大
-   大概率事件：信息量小

可以看出概率与信息量是有相关性的，那么我们可以用概率来计算一个事件的信息量：
$$
h(x)=-\log(p(x))
$$

# 信息熵

我们知道了单一事件的信息量的计算方法，那么对于一个由一系列事件组成的统计分布来说，它的任意随机变量包含的信息称之为信息熵（information entropy）:
$$
\begin{equation} \nonumber
\begin{aligned}
H(x) &= \mathbb{E}[-\log(p(x))] \\\\
     &= - \sum_{x \in X} p(x) \cdot \log(p(x))
\end{aligned}
\end{equation}
$$
从它的计算公式可以看出，信息熵的物理意义其实就是“*一个统计分布中每个变量所包含的信息量的期望值*”，或者更通俗的说法就是一个统计系统中每个事件信息量的平均值。

>   注意 $\log$ 函数如果是以 $\log_2$ 的话，其单位为 *bits*，如果是 $\log_e$ 的话，单位为 *nats*。

前面我们讨论了单一事件的信息量，对于一个系统来说，不同的概率分布会对其信息熵有什么影响呢？我们这里考虑最简单的情况，只有两个事件的统计系统，比如抛硬币。

先不说废话，直接用代码画出不同概率分布情况下的信息熵：

```python
# compare probability distributions vs entropy
from math import log2
from matplotlib import pyplot
 
# calculate entropy
def entropy(events, ets=1e-15):
	return -sum([p * log2(p + ets) for p in events])
 
# define probabilities
probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# create probability distribution
dists = [[p, 1.0 - p] for p in probs]
# calculate entropy for each distribution
ents = [entropy(d) for d in dists]
# plot probability distribution vs entropy
pyplot.plot(probs, ents, marker='.')
pyplot.title('Probability Distribution vs Entropy')
pyplot.xticks(probs, [str(d) for d in dists])
pyplot.xlabel('Probability Distribution')
pyplot.ylabel('Entropy (bits)')
pyplot.show()
```

<img src="https://machinelearningmastery.com/wp-content/uploads/2019/10/Plot-of-Probability-Distribution-vs-Entropy-1024x768.png" style="zoom:50%;" />

我们可以看到，当正反两面的概率相等的时候信息熵最大，而概率越是偏向某一事件的时候，信息熵就越小：

-   Skewed Probability Distribution：低信息熵
-   Balanced Probability Distribution：高信息熵

其实这一点与单一事件的信息量的性质也是一致的，因为当概率分布偏向某一事件的时候，说明该系统存在一个主导事件，该事件概率很大，而信息量很低，造成整个概率分布的信息熵较低。

我们现在已经知道了如何量化一个统计分布的信息熵的方法了，但是我们还是没办法知道哪个模型是更好的模型。我们还需要将两个模型与真实的蠕虫牙齿数量分布进行对于，看看哪个模型的信息损失更低，即我们要计算两个概率分布的差别。这里我们介绍两个工具：交叉熵和 KL 散度。

# 交叉熵

假设有两个概率分布 $p$ 和 $q$，他们之间的差别为：
$$
H(p, q) = - \sum_{x \in X} p(x)\cdot \log(q(x))
$$
我们将这种计算方法称之为交叉熵。

# KL 散度

假设有两个概率分布 $p$ 和 $q$，他们之间的差别为：
$$
\begin{equation} \nonumber
\begin{aligned}
D(p||q) &= \sum_{x \in X} p(x) \cdot (\log p(x) - \log q(x)) \\\\
        &= \sum_{x \in X} p(x) \cdot \log (\frac{p(x)}{q(x)})
\end{aligned}
\end{equation}
$$
到此时，我们就可以计算到底是哪种模型更加符合实际分布了。比如如果我们用 KL 散度的话，得到：
$$
D(\text{observed||Uniform}) = 0.338 \\\\
D(\text{observed||Binomial}) = 0.447
$$
所以使用均匀分布的模型损失的信息更小。

# 交叉熵与 KL 散度的应用

我们在上面通过一个例子简单介绍了交叉熵和 KL 散度，那么他们有什么用呢？交叉熵被广泛用于分类模型的损失函数。

-   $p(x)$：模型的预测输出；
-   $q(x)$：训练数据的标签。

$$
H(p, q) = - (p(\text{class0}) * \log (q(\text{class0})) + p(\text{class1})* \log(q(\text{class1})))
$$

例：

```python
from math import log

q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

def corss_entropy(p, q):
    # 为了防止 log 定义域为 0 的情况
    # p 是训练数据标签， q 是预测值
    return -sum([p[i]*log(q[i]) for i in range(len(p))])

results = []
for i in range(len(p)):
	# calculate cross entropy for the two events
	ce = cross_entropy(p, q)
	print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))
	results.append(ce)
```

```python
>[y=1.0, yhat=0.8] ce: 1.685 nats
>[y=1.0, yhat=0.9] ce: 1.685 nats
>[y=1.0, yhat=0.9] ce: 1.685 nats
>[y=1.0, yhat=0.6] ce: 1.685 nats
>[y=1.0, yhat=0.8] ce: 1.685 nats
>[y=0.0, yhat=0.1] ce: 1.685 nats
>[y=0.0, yhat=0.4] ce: 1.685 nats
>[y=0.0, yhat=0.2] ce: 1.685 nats
>[y=0.0, yhat=0.1] ce: 1.685 nats
>[y=0.0, yhat=0.3] ce: 1.685 nats
```

同样的 KL 散度也可以用于分类模型的损失函数。
$$
D(p, q) = H(p, q) = - (p(\text{class0}) * \log (\frac{p(\text{class0})}{q(\text{class0})}) + p(\text{class1})* \log(\frac{p(\text{class1})}{q(\text{class1})}))
$$
例：

```python
from math import log

q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

def kl_divergence(p, q):
    # 为了防止 log 定义域为 0 的情况
    # p 是训练数据标签+10e-5， q 是预测值
    return sum([p[i]*log((p[i]+10e-5)/q[i]) for i in range(len(p))])

results = []
for i in range(len(p)):
	# calculate cross entropy for the two events
	kl = kl_divergence(p, q)
	print('>[y=%.1f, yhat=%.1f] kl: %.3f nats' % (p[i], q[i], ce))
	results.append(kl)
```

```python
>[y=1.0, yhat=0.8] kl: 1.168 nats
>[y=1.0, yhat=0.9] kl: 1.168 nats
>[y=1.0, yhat=0.9] kl: 1.168 nats
>[y=1.0, yhat=0.6] kl: 1.168 nats
>[y=1.0, yhat=0.8] kl: 1.168 nats
>[y=0.0, yhat=0.1] kl: 1.168 nats
>[y=0.0, yhat=0.4] kl: 1.168 nats
>[y=0.0, yhat=0.2] kl: 1.168 nats
>[y=0.0, yhat=0.1] kl: 1.168 nats
>[y=0.0, yhat=0.3] kl: 1.168 nats
```

# 交叉熵 VS. KL 散度

交叉熵：
$$
H(p, q) = - \sum_{x \in X} p(x)\cdot \log(q(x))
$$
KL 散度：
$$
D(p||q) = \sum_{x \in X} p(x) \cdot \log (\frac{p(x)}{q(x)})
$$

$$
\begin{equation} \nonumber
\begin{aligned}
H(p, q) - D(p || q) &= - \sum_{x \in X} p(x)\cdot \log(q(x)) - \sum_{x \in X} p(x) \cdot \log (\frac{p(x)}{q(x)}) \\\\
                    &= - \sum_{x \in X} p(x) \cdot [\log(q(x)) + \log (\frac{p(x)}{q(x)} ) \\\\
                    &= - \sum_{x \in X} p(x) \cdot \log(q(x) \cdot \frac{p(x)}{q(x)}) \\\\
                    &= - \sum_{x \in X} p(x) \cdot \log(p(x))
\end{aligned}
\end{equation}
$$

即
$$
H(p, q) = D(p ||q) + h(p)
$$
当 $h(p)$ 是常数的时候，交叉熵和 KL 散度是等效的。那么什么时候 $h(p)$ 是常数呢？在分类任务中，我们希望
$$
p(model) \approx p(D) \approx p(truth)
$$
$p(model)$ 表示模型输出，$p(D)$ 表示数据集的分布，$p(truth)$ 表示真实世界的数据分布。当使用 KL 散度作为损失函数的时候，我们是最小化 $D(p(D)||p(model))$，通常情况下 $p(D)$ 是固定不变的。所以，在通常情况下交叉熵损失函数和 KL 散度损失函数是等效的。

## 实验

我们用 keras 在 cifar10 数据集上训练一个 ConvNet 模型进行一下验证：

```python
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Model configuration
img_width, img_height         = 32, 32
batch_size                    = 250
no_epochs                     = 25
no_classes                    = 10
validation_split              = 0.2
verbosity                     = 1

# Load CIFAR10 dataset
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0],3, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 3, img_width, img_height)
    input_shape = (3, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 3)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 3)
    input_shape = (img_width  , img_height, 3)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data.
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = keras.utils.to_categorical(target_train, no_classes)
target_test = keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

loss = keras.losses.kullback_leibler_divergence
# loss = keras.losses.categorical_crossentropy

# Compile the model
model.compile(loss=loss,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split
)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

我们的得到的结果如下：

![](https://www.machinecurve.com/wp-content/uploads/2019/12/kld4.png)

![](https://www.machinecurve.com/wp-content/uploads/2019/12/kld3.png)

实验结果与我们的分析是一致的。

# Not distance

我们介绍交叉熵和 KL 散度时说，这两个量是评估两个分布的差别。通常情况下，我们说两个东西的差别我们通常用距离来表示，比如欧氏距离，余弦距离等等。那么交叉熵和 KL 散度也是两个分布的距离吗？答案是 **No**!

**距离** 是一个没有方向性的概念， A 到 B 的距离是 10，那么 B 到 A 的距离也应该是 10，这叫做对称性。但是交叉熵和 KL 散度是不具有对称性的，即
$$
H(p, q) \ne H(q, p) \\\\
D(p||q) \ne D(q||p)
$$
这一点我们可以用上面的例子就可以证明。所以无论是交叉熵还是 KL 散度都不是距离。

# Jenson-Shannon 散度

JS 散度是另一张计算两个分布之间差距的方法：
$$
JS(p||q) = \frac{1}{2} D(p||m) + \frac{1}{2} D(q||m)
$$
其中 $m=1/2(p+q)$。

```python
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

很明显 JS 散度是具有对称性的，所以我们可以将 JS 散度单程两个分布的距离。

# Reference

1.   [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained). Count Yayesie. 2017
2.   [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/).  Jason Brownlee. 2019
3.   [How to Calculate the KL Divergence for Machine Learning](https://machinelearningmastery.com/divergence-between-probability-distributions/).  Jason Brownlee. 2019
4.   [how to use kullback leibler divergence kl divergence with keras](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-kullback-leibler-divergence-kl-divergence-with-keras.md). christianversloot. 
