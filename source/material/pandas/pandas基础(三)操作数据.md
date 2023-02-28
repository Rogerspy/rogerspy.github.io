<h1>Pandas基础教程（三）数据操作</h1>

[TOC]

`pandas`是一种基于`Numpy`的数据工具，因此它能很好的支持`Numpy`的通用函数。但是与`Numpy`之间也存在些许不同。

对于一元操作（比如三角函数），Pandas会在输出中保留索引名和列名；对于二元操作（比如加减乘除），Pandas会自动排列索引。这就意味着如果两列数据索引不同，就会存在数据“不对齐”的情况。

# 1. 索引存留

```python
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser
#> 0    6
#> 1    3
#> 2    7
#> 3    4
#> dtype: int64

df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
df
#>  	A 	B 	C 	D
#> 0 	6 	9 	2 	6
#> 1 	7 	4 	3 	7
#> 2 	7 	2 	5 	4
```
如果我们应用通用函数在这两个pandas对象上面会得到另外的两个pandas对象，保留原有的索引名和列名。

```python
np.exp(ser)
#> 0     403.428793
#> 1      20.085537
#> 2    1096.633158
#> 3      54.598150
#> dtype: float64

np.sin(df * np.pi / 4)
#> 	A 	B 	C 	D
#> 0 	-1.000000 	7.071068e-01 	1.000000 	-1.000000e+00
#> 1 	-0.707107 	1.224647e-16 	0.707107 	-7.071068e-01
#> 2 	-0.707107 	1.000000e+00 	-0.707107 	1.224647e-16
```

# 2. 索引排列

```python
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
```

```python
population / area

#> Alaska              NaN
#> California    90.413926
#> New York            NaN
#> Texas         38.018740
#> dtype: float64
```

Pandas会在相同的索引位置进行除法，而如果没有相同的索引，对应的位置会标记为缺失值，比如`NaN`。

如果我们不希望存在缺失值，我们可以对缺失值进行填充。具体方法下一节进行介绍。

## 2.1 `DataFrame`中的数据排列

```python
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
A
#>  	A 	B
#> 0 	1 	11
#> 1 	5 	1

B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
B
#> 	B 	A 	C
#> 0 	4 	0 	9
#> 1 	5 	8 	0
#> 2 	9 	2 	6
```

```python
A + B

#>  	A 	B 	C
#> 0 	1.0 	15.0 	NaN
#> 1 	13.0 	6.0 	NaN
#> 2 	NaN 	NaN 	NaN
```

下表中列出了Python内置的操作符和对应的Pandas方法caozuo

|Python操作符|Pandas方法|
|-|-|
|+|add()|
|-|sub(), subtract()|
|*|mul(), multiply()|
|/|truediv(), div(), divide()|
|//|floordiv()|
|%|mod()|
|**|pow()|

## 2.2 `DataFrame`与`Series`之间的操作

`DataFrame`与`Series`之间的操作类似于Numpy的二维数组和一维数组之间的操作。不同的是pandas会保留索引和列名，这就意味着pandas会保留数据的上下文，没有相应数据的位置会标记为缺失值。

- **Numpy:**

```python
A = np.random.randint(10, size=(3, 4))
A

#> array([[3, 8, 2, 4],
#>        [2, 6, 4, 8],
#>        [6, 1, 3, 8]])
```

```python
A - A[0]

#> array([[ 0,  0,  0,  0],
#>        [-1, -2,  2,  4],
#>        [ 3, -7,  1,  4]])
```

- **Pandas:**

```python
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]

#>  	Q 	R 	S 	T
#> 0 	0 	0 	0 	0
#> 1 	-1 	-2 	2 	4
#> 2 	3 	-7 	1 	4
```

```python
df.subtract(df['R'], axis=0)

#> 	Q 	R 	S 	T
#> 0 	-5 	0 	-6 	-4
#> 1 	-4 	0 	-2 	2
#> 2 	5 	0 	2 	7
```

```python
halfrow = df.iloc[0, ::2]
halfrow
#> Q    3
#> S    2
#> Name: 0, dtype: int64

df - halfrow

#> 	Q 	R 	S 	T
#> 0 	0.0 	NaN 	0.0 	NaN
#> 1 	-1.0 	NaN 	2.0 	NaN
#> 2 	3.0 	NaN 	1.0 	NaN
```
