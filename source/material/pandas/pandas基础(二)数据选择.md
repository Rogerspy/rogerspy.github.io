<h1>Pandas基础教程（二）数据选择</h1>

[TOC]

# 1. `Series`中数据选择

## 1.1 `Series`作为字典

就像一个字典，`Series`可以通过键/值方式来选择数据。

```python
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data
#> a    0.25
#> b    0.50
#> c    0.75
#> d    1.00
#> dtype: float64

data['b']
#> 0.5
```

我们也可以用类似字典的方法来检查键/索引和值。

```python
'a' in data
#> True

data.keys()
#> Index(['a', 'b', 'c', 'd'], dtype='object')

list(data.items())
#> [('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)]
```

还可以通过与字典类似的语法来修改数据。

```python
data['e'] = 1.25
data

#> a    0.25
#> b    0.50
#> c    0.75
#> d    1.00
#> e    1.25
#> dtype: float64
```

## 1.2 `Series`作为一维数组

`Series`还提供了类似Numpy array风格的数据选择方式——切片(slicing)、掩膜(masking)、花式索引(fancy idnexing)。

```python
# 通过index值来切片
data['a':'c']
#> a    0.25
#> b    0.50
#> c    0.75
#> dtype: float64

# 通过位置索引来切片
data[0:2]
#> a    0.25
#> b    0.50
#> dtype: float64

# 掩膜
data[(data > 0.3) & (data < 0.8)]
#> b    0.50
#> c    0.75
#> dtype: float64

# 花式索引
data[['a', 'e']]
#> a    0.25
#> e    1.25
#> dtype: float64
```

## 1.3 索引器：`loc`, `iloc`和`ix`

前面我们介绍的切片和索引的方法很多时候会引起混淆。比如如果你有一个使用整数作为索引值的`Series`，当进行索引操作时`data[1]`表示利用索引值进行索引，而切片时`data[1:3]`表示使用位置索引进行切片。代码如下：

```python
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
#> 1    a
#> 3    b
#> 5    c
#> dtype: object

# 利用索引值进行索引
data[1]
#> 'a'

# 利用索引位置进行切片
data[1:3]
#> 3    b
#> 5    c
#> dtype: object
```

可以看到`data[1:3]`并没有得到`index　=　1和3`这个两个索引的值，而是得到１和３这两个位置的值。这样就容易引起混淆。为了避免这种混淆，`Pandas`提供了集中索引方法——`loc`、`iloc`和`ix`。

- **`loc`**

`loc`总是以索引值来进行索引和切片

```python
data.loc[1]
#> 'a'

data.loc[1:3]
#> 1    a
#> 3    b
#> dtype: object
```

- **`iloc`**

`iloc`总是以索引的位置进行索引和切片

```python
data.iloc[1]
#> 'b'

data.iloc[1:3]
#> 3    b
#> 5    c
#> dtype: object
```
- **`ix`**

`ix`是一种混合的索引方式。对`Series`来说,如果索引值是字符串就等效于`iloc`，如果索引值是整数，就相当于`loc`。

```python
data.ix[1]
#> 'a'

data.ix[1:3]
#> 1    a
#> 3    b
#> dtype: object
```

# 2. `DataFrame`中数据选择

## 2.1 `DataFrame`作为字典

```python
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data

#>               area       pop
#> California  423967  38332521
#> Florida     170312  19552860
#> Illinois    149995  12882135
#> New York    141297  19651127
#> Texas       695662  26448193
```

```python
data['area']
#> California    423967
#> Florida       170312
#> Illinois      149995
#> New York      141297
#> Texas         695662
#> Name: area, dtype: int64

data.area
#> California    423967
#> Florida       170312
#> Illinois      149995
#> New York      141297
#> Texas         695662
#> Name: area, dtype: int64

# 上述两种方法获得的是相同的对象
data.area is data['area']
#> True
```

上面两种方法并不是一直都等价。比如如果列名不是字符串或者列名与`DataFrame`的方法名冲突的时候。

```python
data.pop is data['pop']
#> False
```

还可以用类似字典的语法修改对象，比如增加一列：

```python
data['density'] = data['pop'] / data['area']
data

#>               area       pop     density
#> California  423967  38332521   90.413926
#> Florida     170312  19552860  114.806121
#> Illinois    149995  12882135   85.883763
#> New York    141297  19651127  139.076746
#> Texas       695662  26448193   38.018740
```

## 2.2 `DataFrame`作为二维数组

```python
data.values

#> array([[  4.23967000e+05,   3.83325210e+07,   9.04139261e+01],
#>        [  1.70312000e+05,   1.95528600e+07,   1.14806121e+02],
#>        [  1.49995000e+05,   1.28821350e+07,   8.58837628e+01],
#>        [  1.41297000e+05,   1.96511270e+07,   1.39076746e+02],
#>        [  6.95662000e+05,   2.64481930e+07,   3.80187404e+01]])
```

```python
data.T
#>            California       Florida      Illinois      New York         Texas
#> area     4.239670e+05  1.703120e+05  1.499950e+05  1.412970e+05  6.956620e+05
#> pop      3.833252e+07  1.955286e+07  1.288214e+07  1.965113e+07  2.644819e+07
#> density  9.041393e+01  1.148061e+02  8.588376e+01  1.390767e+02  3.801874e+01

data.values[0]
#> array([  4.23967000e+05,   3.83325210e+07,   9.04139261e+01])

data.iloc[:3, :2]
#>               area       pop
#> California  423967  38332521
#> Florida     170312  19552860
#> Illinois    149995  12882135

data.loc[:'Illinois', :'pop']
#>               area       pop
#> California  423967  38332521
#> Florida     170312  19552860
#> Illinois    149995  12882135

data.ix[:3, :'pop']
#>               area       pop
#> California  423967  38332521
#> Florida     170312  19552860
#> Illinois    149995  12882135
```

```python
data.loc[data.density > 100, ['pop', 'density']]
#>                pop     density
#> Florida   19552860  114.806121
#> New York  19651127  139.076746

data[data.density > 100]
#>             area       pop     density
#> Florida   170312  19552860  114.806121
#> New York  141297  19651127  139.076746
```
