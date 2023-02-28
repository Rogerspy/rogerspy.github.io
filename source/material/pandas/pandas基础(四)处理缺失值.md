<h1>Pandas基础教程（四）处理缺失值</h1>

[TOC]

# 1. Pandas中的缺失值

Pandas中的缺失值有两种表示方式：特殊的浮点值`NaN`和python中的`None`对象。

## 1.1 `None`

```python
import numpy as np
import pandas as pd

vals1 = np.array([1, None, 3, 4])
vals1

#> array([1, None, 3, 4], dtype=object)
```

`dtype=object`说明最适合这个数组的类型是一个python对象。虽然在某些时候很有用，但是在数据上的任何操作都会在python层面完成，其开销要比使用原生类型的数组大得多。

```python
for dtype in ['object', 'int']:
    print("dtype =", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()

#> dtype = object
#> 10 loops, best of 3: 70.7 ms per loop
#>
#> dtype = int
#> The slowest run took 9.64 times longer than the fastest. This could mean that an intermediate result is being cached.
#> 100 loops, best of 3: 2.6 ms per loop
```

并且在一些聚合(aggregations)操作上会报错：

```python
vals1.sum()

#> TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
```

说明`None`不能与整数相加。类似的加减乘除之类的操作都不可以。

## 1.2 `NaN`

另一个表示缺失数据的值是`NaN`(Not a Number)，与`None`不同的是，`NaN`是一种特殊的浮点型数据（根据标准的IEEE浮点表达，所有系统都把`NaN`当做浮点型处理）。

```python
vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

#> dtype('float64')
```

Numpy选择使用`NaN`来表达缺失值，它支持更快的数据操作，并且当`NaN`与其他类型的数据进行操作的时候，无论是什么样的操作，最后的结果都是`NaN`而不会报错。

```python
1 + np.nan
#> NaN

0 *  np.nan
#> NaN

vals2.sum(), vals2.min(), vals2.max()
#> (NaN, NaN, NaN)
```

Numpy提供了一些特殊的聚合操作来避免`NaN`的影响

```python
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

#> (8.0, 1.0, 4.0)
```

记住，`NaN`是一种特殊的浮点型，没有等效的整数，字符串或其他类型。

```python
np.nan == np.nan
#> False

np.isnan(np.nan)
#> True
```

## 1.3 pandas中的`None`和`NaN`

```python
pd.Series([1, np.nan, 2, None])

#> 0    1.0
#> 1    NaN
#> 2    2.0
#> 3    NaN
#> dtype: float64
```

可以看到pandas会自动把`None`转换成`NaN`,并且虽然１和２是`int`型数据，但`NaN`是浮点型，目前没有与整型对应的缺失值表示，所以pandas会把数据转换成浮点型。（据说pandas正在考虑以`NA`来表示整型缺失值，但到目前为止还没有）

那如果我们想保留`None`不被替换掉，可以使`dtype='object'`,如下：

```python
pd.Series([1, np.nan, 2, None], dtype='object')

#> 0    1.0
#> 1    NaN
#> 2    2.0
#> 3    None
#> dtype: object
```

下表中列出了当数据中存在缺失值的时候，数据类型的自动转换

|类型|值|
|-|-|
|float -> float|np.nan|
|object -> object|None, np.nan|
|int -> float|np.nan|
|boolean -> object|None, np.nan|

# 2. 处理缺失值

pandas提供了一些方法用以检测，删除或替换缺失值。

- `isnull()`：生成一个mask，表示每个位置是否是缺失值。
- `notnull()`：与`isnull`相反。
- `dropna()`：返回过滤掉缺失值的数据。
- `fillna()`：返回一个填充了缺失值的数据。

## 2.1 检测缺失值

```python
data = pd.Series([1, np.nan, 'hello', None])

data.isnull()
#> 0    Falsede
#> 1     True
#> 2    False
#> 3     True
#> dtype: bool

data.notnull()
#> 0     True
#> 1    False
#> 2     True
#> 3    False
#> dtype: bool
```

在前一章索引中我们提到，布尔型的mask可以用来直接作为pandas数据的索引来取值：

```python
data[data.notnull()]
#> 0        1
#> 2    hello
#> dtype: object
```

## 2.2 删除缺失值

pandas提供了一种删除缺失值的方法——`dropna()`。需要注意的是，对`DataFrame`来说，不能删除单个值，只能删除包含缺失值的行或者列。

```python
data.dropna()
#> 0        1
#> 2    hello
#> dtype: object

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df
#>  	0 	1 	2
#> 0 	1.0 	NaN 	2
#> 1 	2.0 	3.0 	5
#> 2 	NaN 	4.0 	6

df.dropna()

#> 	0 	1 	2
#> 1 	2.0 	3.0 	5
```

默认是删除行(`axis=0`或`axis='rows'`)，也可设置`axis=1`(或`axis='columns'`)删除列。

```python
df.dropna(axis='columns')

#>  	2
#> 0 	2
#> 1 	5
#> 2 	6
```

有时候我们只想删除那些整列或整行都是缺失值的数据，或者更个性化的设置。`dropna()`提供了两个参数——`how`或者`thresh`。默认`how=any`表示，一行或者一列中有任意个缺失值，该行或者该列就会被删除；还可以设置`how=all`表示，整行或者整列都是缺失值才会被删除。`thresh`表示行或者列至少有多少个非缺失值才会被保留。

```python
df[3] = np.nan
df
#> 	0 	1 	2 	3
#> 0 	1.0 	NaN 	2 	NaN
#> 1 	2.0 	3.0 	5 	NaN
#> 2 	NaN 	4.0 	6 	NaN

df.dropna(axis='columns', how='all')
#> 	0 	1 	2
#> 0 	1.0 	NaN 	2
#> 1 	2.0 	3.0 	5
#> 2 	NaN 	4.0 	6

df.dropna(axis='rows', thresh=3)
#>　	0 	1 	2 	3
#> 1 　2.0 	3.0 	5 	NaN
```

## 2.3 填充缺失值

有时候我们并不想删除缺失值，而是想用一些值来对其进行填充。pandas提供了`fillna()`方法来帮我们实现。

```python
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
#> a    1.0
#> b    NaN
#> c    2.0
#> d    NaN
#> e    3.0
#> dtype: float64

data.fillna(0)
#> a    1.0
#> b    0.0
#> c    2.0
#> d    0.0
#> e    3.0
#> dtype: float64
```

不止可以填充固定值，还可以根据缺失值前后的数据进行填充。

```python
# 前面的数据
data.fillna(method='ffill')
#> a    1.0
#> b    1.0
#> c    2.0
#> d    2.0
#> e    3.0
#> dtype: float64

# 后面的数据
data.fillna(method='bfill')
#> a    1.0
#> b    2.0
#> c    2.0
#> d    3.0
#> e    3.0
#> dtype: float64
```

对`DataFrame`来说，也是相似的，只不过我们还可以设置`axis`参数。

```python
df.fillna(method='ffill', axis=1)

#> 	0 	1 	2 	3
#> 0 	1.0 	1.0 	2.0 	2.0
#> 1 	2.0 	3.0 	5.0 	5.0
#> 2 	NaN 	4.0 	6.0 	6.0
```

需要注意的是，使用前/后数据填充时，如果前后数据不可用，则仍保留缺失值。
