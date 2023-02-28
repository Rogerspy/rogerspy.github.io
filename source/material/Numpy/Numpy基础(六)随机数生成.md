<h1>Numpy基础教程（六）随机数生成</h1>

[TOC]

# 常用随机数相关函数

- `seed`：随机数生成器的种子。

```python
np.random.seed()
print("random number with default seed： ", np.random.random())
#random number with default seed：  0.026927763885101763

np.random.seed(10)
print("random number with int seed： ", np.random.random())
#random number with int seed：  0.771320643266746

np.random.seed(10)
print("random number with int seed： ", np.random.random())
#random number with int seed：  0.771320643266746
```

- `permutation`：随机排列一个序列或返回一个随机排列范围。

```python
np.random.permutation(10)
#> array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

arr = np.arange(9).reshape((3, 3))
np.random.permutation(arr)
#> array([[6, 7, 8],
#>        [0, 1, 2],
#>        [3, 4, 5]])
```

- `shuffle`：对一个序列就地随机排列。

```python
arr = np.arange(10)
np.random.shuffle(arr)
arr
#> [1 7 5 2 9 4 3 6 0 8]

# 多维数组只会沿着第一个轴进行随机排列
arr = np.arange(9).reshape((3, 3))
np.random.shuffle(arr)
arr
#> array([[3, 4, 5],
#>        [6, 7, 8],
#>        [0, 1, 2]])
```

- `rand`：产生均匀分布的数组。

```python
np.random.rand(3,2)
#> array([[ 0.14022471,  0.96360618],  
#>        [ 0.37601032,  0.25528411],  
#>        [ 0.49313049,  0.94909878]])
```

- `randint`：从给定范围内随机选取整数。

```python
np.random.randint(2, size=10)
#> random number with int seed：  0.771320643266746

np.random.randint(5, size=(2, 4))
#> array([[4, 0, 2, 1],
#>        [3, 2, 2, 0]])
```

- `randn`：产生平均值为０，标准差为１的正态分布。

```python
np.random.randn()
#>2.1923875335537315

2.5 * np.random.randn(2, 4) + 3
#> array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  
#>        [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])
```

- `binomial`：产生二项分布。

$$
P(N) = \binom{n}{N}p^N(1-p)^{n-N}
$$

```python
np.random.binomial(10,　0.5)　#抛10次硬币，每次正面概率0.5
#> 6 # 正面出现的次数

np.random.binomial(10,0.5, 1000)
# 抛10次硬币，每次正面概率0.5，这个过程再重复1000次
```

- `normal`：产生正态分布。

$$
p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} }
$$

```python
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 0.1 # 平均值与标准差
s = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()
```

- `beta`：产生Beta分布。

$$
f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}(1 - x)^{\beta - 1}
$$

其中

$$
B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}(1 - t)^{\beta - 1} dt
$$

```python
np.random.beta(0.5,0.5,10)
#> array([3.50185726e-01, 8.25361480e-01, 8.45937925e-01, 2.66085065e-04,
#>        1.93886404e-01, 7.83191352e-01, 9.74305512e-02,q 6.53044453e-01,
#>        8.89779868e-01, 5.42610759e-01])
```

- `chisquare`：产生卡方分布。

$$
Q = \sum_{i=0}^{\mathtt{df}} X^2_i \rightarrow Q \sim \chi^2_k \\\\
p(x) = \frac{(1/2)^{k/2}}{\Gamma(k/2)}x^{k/2 - 1} e^{-x/2}
$$



其中
$$
\Gamma(x) = \int_0^{-\infty} t^{x - 1} e^{-t} dt
$$

```python
np.random.chisquare(2,4)

#> array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272])
```

- `gamma`：产生Gamma分布。

$$
p(x) =x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)}
$$

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)

count, bins, ignored = plt.hist(s, 50, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))

plt.plot(bins, y, linewidth=2, color='r')
plt.show()
```

- `uniform`：产生[0, 1)的均匀分布。

$$
p(x) = \frac{1}{b - a}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

s = np.random.uniform(-1,0,1000)

count, bins, ignored = plt.hist(s, 15, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()
```
