<h1>Pandas基础教程（一）数据结构</h1>

[TOC]

基本上我们可以认为Pandas是一个加强版的Numpy，它的行和列可以用特定的标签来表示，而不仅仅是用数字作为索引。Pandas基于基本的数据结构提供了一系列非常有用的工具，方法和函数。从本章开始我们将学习Pandas，首先学习的是Pandas的三种基本数据结构：`Series`, `DataFrame`, `Index`。

# 1. `Series`对象

通常我们会和Numpy一起使用Pandas，首先导入模块：

```python
import numpy as np
import pandas as pd
```

`Series`对象是一维的索引化的数组。我们可以通过两种方法来构建`Series`：数组和字典。

## 1.1 通过数组构建`Series`

```python
data = pd.Series([1,2,3,4,5])
data

#> 0    1
#> 1    2
#> 2    3
#> 3    4
#> 4    5
#> dtype: int64
```

上面的输出结果左边一列表示索引(index)，右边一列表示数据(values)。我们可以通过给index赋值来设置索引名：

```python
data = pd.Series([1,2,3,4,5], index=['a', 'b', 'c', 'd', 'e'])
data

#> a    1
#> b    2
#> c    3
#> d    4
#> e    5
```

当我们省略index参数的时候，index会默认是从0到n-1的数组，其中n是数组长度。

我们来看一下一种特殊情况，数据长度和索引长度不一致会发生什么？

```python
data = pd.Series(5, index=['a', 'b', 'c'])
data
#> a    5
#> b    5
#> c    5
#> dtype: int64

data = pd.Series([4, 5], index=['a', 'b', 'c'])
data
#> ValueError: Wrong number of items passed 2, placement implies 3

data = pd.Series([1,2,3], index=['a'])
data
#> ValueError: Wrong number of items passed 3, placement implies 1
```

我们可以看到当数据只有一个值的时候（改成只有一个元素的列表也一样），pandas会把数据广播到每一个索引上，如果数据多余１个就会报错；而无论什么情况当数据多余索引的时候都会报错。

## 1.2 通过字典构建`Series`

通过字典来构建`Series`，index默认是经过排序的字典的key。

```python
pd.Series({2:'a', 1:'b', 3:'c'})

#> 1    b
#> 2    a
#> 3    c
#> dtype: object
```

同样，我们来看下特殊情况：

```python
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
#> 3    c
#> 2    a
#> dtype: object

pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2, 1, 0])

#> 3      c
#> 2      a
#> 1      b
#> 0    NaN
#> dtype: object
```

说明通过字典构建`Series`也还可以指定index，并且数据排序与index一致而不会自动排序。另外，数据长度不一致也不会报错。

# 2. `DataFrame`对象

`DataFrame`可以看成是二维的numpy数组，它的行叫做`index`,列叫做`column`。行和列的索引都可以命名。

构建`DataFrame`的方法通常也是有两种：数组和字典。

## 2.1 通过数组构建`DataFrame`

- 二维数组

```python
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])

#>         foo       bar
#> a  0.664491  0.349118
#> b  0.221553  0.466591
#> c  0.654042  0.185945
```

- 结构化的数组[^1]

[^1]: [Structured Data: NumPy's Structured Arrays](https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html)

```python
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A
#> array([(0, 0.0), (0, 0.0), (0, 0.0)],
#>       dtype=[('A', '<i8'), ('B', '<f8')])

pd.DataFrame(A)
#>    A    B
#> 0  0  0.0
#> 1  0  0.0
#> 2  0  0.0
```

## 2.2 通过字典构建`DataFrame`

- 字典列表

```python
data = {'a':[0, 1, 2], 'b':[2, 4, 6]}
pd.DataFrame(data)
#>    a  b
#> 0  0  2
#> 1  1  4
#> 2  2  6


data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
pd.DataFrame(data)

#>    a  b
#> 0  0  0
#> 1  1  2
#> 2  2  4
```

甚至字典中的某些`key`丢失，Pandas也会自动用NaN填充空缺。

```python
pd.DataFrame([{'a':1, 'b':2},
              {'b':3, 'c':4}])

#>      a  b  c
#> 0  1.0  2  NaN
#> 1  NaN  3  4.0
```
- `Series`

```python
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population

#> California    38332521
#> Florida       19552860
#> Illinois      12882135
#> New York      19651127
#> Texas         26448193
#> dtype: int64

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area

#> California    423967
#> Florida       170312
#> Illinois      149995
#> New York      141297
#> Texas         695662
#> dtype: int64

pd.DataFrame({'population':population,'area':area})

#>               area  population
#> California  423967    38332521
#> Florida     170312    19552860
#> Illinois    149995    12882135
#> New York    141297    19651127
#> Texas       695662    26448193
```

# 3. `Index`对象

`Index`可以看成是一种特殊的不可变的数组或者经过排序的集合（**`Index`可以包含重复值**）。

```python
ind = pd.Index([2, 3, 5, 7, 11])
ind

#> Int64Index([2, 3, 5, 7, 11], dtype='int64')
```

## 3.1 不可变数组

```python
ind[1]
#> 3

ind[::2]
#> Int64Index([2, 5, 11], dtype='int64')

print(ind.size, ind.shape, ind.ndim, ind.dtype)
#> 5 (5,) 1 int64

ind[1] = 0
# 报错
#> TypeError: Index does not support mutable operations
```

## 3.2 有序集合

```python
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 2, 5, 7, 11])

# 交集
indA & indB
#> Int64Index([5, 7], dtype='int64')

# 并集
indA | indB
#> Int64Index([1, 2, 2, 3, 5, 7, 9, 11], dtype='int64')

# 对等差分（symmetric difference）
indA ^ indB
#> Int64Index([1, 2, 3, 9, 11], dtype='int64')
```
