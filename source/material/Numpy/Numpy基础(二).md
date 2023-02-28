<h1>Numpy基础教程（二）通用函数：快速元素级数组函数</h1>

[TOC]

通用函数（ufunc）是一种对ndarray中数据进行元素级运算的函数。可以将其看成简单函数的矢量化包装器。

# 1. 一元ufunc

- `abs`, `fabs`：计算整数，浮点数或复数的绝对值。对于非复数可以使用fabs，速度更快。

```python
arr = np.array([-1, 2, 3])
np.abs(arr)

#> array([1, 2, 3])
```
- `sqrt`：计算各元素的平方根。

```python
np.sqrt(4)

#> 2.0
```
- `square`：计算各元素的平方。

```python
np.square(2)

#> 4
```

- `exp`：计算个元素的指数$$e^x$$。
```python
np.exp(2)

#> 7.38905609893065
```

- `log`, `log10`, `log2`, `log1p`：分别为自然对数，底数为10的对数，底数为2的对数以及以e为底的log(1+x)。
```python
np.log(np.e)
#> 1.0

np.log10(10)
#> 1.0

np.log2(2)
#> 1.0

np.log1p(np.e - 1)
#> 1.0
```

- `sign`：计算个元素的正负号：１（正数），０（零），-１（负数）。

```python
np.sign([-5., 4.5, 0., 5-2j])

#> array([-1., 1., 0., 1.+0.j])
```

- `ceil`：计算大于等各元素的最小整数。

```python
np.ceil([-0.1, 1., 1.5])

#> array([-0., 1., 2.])
```
- `floor`：计算小于等于各元素的最大整数。

```python
np.floor([-0.1, 1., 1.5])

#> array([-1., 1., 1.])
```
- `rint`, `round`, `around`：计算各元素值的四舍五入整数(保留dtype)，`round`和`around`等效，可以指定保留小数位数。

```python
np.rint([-0.111, 1.111, 1.555])
np.round([-0.111, 1.111, 1.555], decimals=2)
np.around([-0.111, 1.111, 1.555], decimals=2)

#> array([-0.,  1.,  2.])
#> array([-0.11,  1.11,  1.56])
#> array([-0.11,  1.11,  1.56])
```

**注意：**
```python
np.rint([-0.5, 0.5])

#> array([-0., 0.])
```
这个结果与我们预想的结果不一致，是bug吗？不是的，这是由于根据IEEE标准[^1][^2]浮点数不精确的表示方式所引起的误差导致的。另可参考[14. Floating Point Arithmetic: Issues and Limitations](https://docs.python.org/2/tutorial/floatingpoint.html#tut-fp-issues)以及知乎关于这个问题的讨论：[Python如何将17.955四舍五入保留两位小数？（直接用round结果不对）](https://www.zhihu.com/question/31156619)和[Python 为什么不解决四舍五入(round)的“bug”？](https://www.zhihu.com/question/20128906)


[^1]: "Lecture Notes on the Status of  IEEE 754", William Kahan,
           http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
[^2]: "How Futile are Mindless Assessments of Roundoff in Floating-Point Computation?", William Kahan,
           http://www.cs.berkeley.edu/~wkahan/Mindless.pdf


- `modf`：求数组中的小数和整数部分，以两个数组的形式独立表示。

```python
arr = randn(7) * 5
np.modf(arr)

#> (array([ 0.01258989, -0.2692081 ,  0.45008401, -0.09443583,  0.23231111,
#>         -0.1496405 , -0.14591915]), array([ 3., -2.,  2., -9.,  0., -0., -0.]))
```

- `isnan`：返回一个表示“哪些值是NaN”的布尔型数组。

```python
a = np.array([np.nan, 1., np.nan, 2.1])
np.isnan(a)

#> array([ True, False,  True, False])
```
- `isfinite`, `isinf`：分别返回有穷的和无穷的值的布尔型数组。

```python
a = np.array([np.nan, 1., np.inf, 2.1])
np.isinf(a)
np.isfinite(a)

#> array([False, False,  True, False])
#> array([False,  True, False,  True])
```

- `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh`：普通三角函数和双曲型三角函数。

```python
np.cos([np.pi/3, np.pi/4, np.pi/6])
np.sin([np.pi/3, np.pi/4, np.pi/6])
np.tan([np.pi/3, np.pi/4, np.pi/6])

#> array([0.5       , 0.70710678, 0.8660254 ])
#> array([0.8660254 , 0.70710678, 0.5       ])
#> array([1.73205081, 1.        , 0.57735027])
```

- `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`：反三角函数。

```python
np.arccos(0.5)

#> 1.0471975511965979
```
- `logical_not`, `logical_and`, `logical_or`, `logical_xor`：计算各元素的not, and, or, xor值。

# 2. 二元ufunc

- `floor_divide`：向下圆整除（丢弃余数）。等效于a//b。
```python
a = np.array([4, 5, 6])
b = np.array([1, 2, 3])
np.floor_divide(a, b)

#> array([4, 2, 2])

a // b

#> array([4, 2, 2])
```
- `power`：对第一个数组中的元素计算第二个数组中相应的幂次方，即$$a^b$$。等效于$$a**b$$。

```python
a = np.array([4, 5, 6])
b = np.array([1, 2, 3])

np.power(a, b)

#> array([  4,  25, 216])

a ** b

#> array([  4,  25, 216])
```

- `maximum`, `fmax`, `minimum`, `fmin`：元素级最大值最小值计算，fmax将忽略NaN。

```python
a = np.array([4, 5, 6])
b = np.array([1, 2, 9])

np.maximum(a, b)

#> array([4, 5, 9])
```

- `mod`：　求模运算（余数）。等效于a%b。
```python
a = np.array([4, 5, 6])
b = np.array([1, 2, 9])

np.mod(a, b)

#> array([0, 1, 6])
```

- `copysign`：将第二个数组中的符号复制给第一个数组。

```python
a = np.array([4, 5, 6])
b = np.array([1, -0, -9])

np.copysign(a, b)

#> array([ 4.,  5., -6.])
```

除此以外，加减乘除运算，比较运算，逻辑运算等都可以使用和标量运算相同的符号进行计算。也可以使用numpy函数来计算。运算规则满足Broadcasting规则。详见上一章。
