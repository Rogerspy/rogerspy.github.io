<h1>Pandas基础教程（五）层次化索引</h1>

[TOC]

到目前为止，我们基本上集中在一维和二维数据的操作上，但实际上我们经常遇到更高维的数据。下面我们就来介绍用pandas处理更高维的数据。

实际上pandas也提供了两个数据对象来处理三维和四维数据——`Panel`和`Panel4D`。但这两个对象不如层次化索引常用，因此，我们主要介绍层次化索引，如果对上面两种方法感兴趣可以参考[Further Resources](https://jakevdp.github.io/PythonDataScienceHandbook/03.13-further-resources.html)。

```python
import pandas as pd
import numpy as np
```

# 1. 多索引`Series`

首先让我们考虑一下如何在一维的`Series`中表示二维数据。

## 1.1 不好的方式

假设你想记录每一个地方两年的数据，如何用pandas表示。

```python
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop

#> (California, 2000)    33871648
#> (California, 2010)    37253956
#> (New York, 2000)      18976457
#> (New York, 2010)      19378102
#> (Texas, 2000)         20851820
#> (Texas, 2010)         25145561
#> dtype: int64
```

为什么说这是一种不好的方式呢？假设我们现在要选取其中每个地方2010年的数据:

```python
pop[[i for i in pop.index if i[1] == 2010]]

#> (California, 2010)    37253956
#> (New York, 2010)      19378102
#> (Texas, 2010)         25145561
#> dtype: int64
```

虽然ll我们得到了想要的结果，但代码和逻辑都显得有点乱。如果我们想要简洁明了的代码，多索引(`MultiIndex`)就派上用场了。

## 1.2 一种好的方式：Pandas MultiIndex

```python
index = pd.MultiIndex.from_tuples(index)
index

#> MultiIndex(levels=[['California', 'New York', 'Texas'], [2000, 2010]],
#>            labels=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
```
注意`MultiIndex`中包含多级别的索引(multi levels)（地名和年份）以及多级别的标签(multi labels)。我们再来看一下结果：

```python
pop = pop.reindex(index)
pop

#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

前两列表示multiple index的值，第三列表示数据。需要注意的是，所有空白的地方表示与前相同的index。

现在我们再来获取2010年的数据：

```python
pop[:, 2010]

#> California    37253956
#> New York      19378102
#> Texas         25145561
#> dtype: int64
```

这样看起来就舒服多了，有木有？先别激动，还有更炫酷的操作。

## 1.3 MultiIndex作为额外的维度

仔细观察上面的数据，虽然我们的数据是`Series`，但看起来更像是`DataFrame`。pandas提供了一对操作，可以将MultiIndex的`Series`转换成`DataFrame`——`unstack()`和`stack`。前者是将`Series`转换成`DataFrame`，后者是还原。

```python
pop_df = pop.unstack()
pop_df
#> 　　　　　　　	2000　　 	2010
#> California 	33871648 	37253956
#> New York 	18976457 	19378102
#> Texas 	20851820 	25145561

pop_df.stack()
#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

这组操作看起来美如画，但对我们来说它的实际意义在哪呢？很简单，正如我们开篇所讲，它能用来表示比二维更高维度的数据。

```python
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
pop_df
#>  	        	total    	under18
#> California 	2000 	33871648 	9267089
#>              2010 	37253956 	9284094
#> New York  	2000 	18976457 	4687374
#>              2010 	19378102 	4318033
#> Texas 	2000 	20851820 	5906301
#>              2010 	25145561 	6879014

f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()
#>          	2000    	2010
#> California 	0.273594 	0.249211
#> New York 	0.247010 	0.222831
#> Texas 	0.283251 	0.273568
```

# 2. 创建MultiIndex


最直接的就是传入具有多个索引的列表。

```python
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
df
#> 		data1 	data2
#> a 	1 	0.554233 	0.356072
#>      2 	0.925244 	0.219474
#> b 	1 	0.441759 	0.610054
#>      2 	0.171495 	0.886688
```

类似的，还可以传入字典：

```python
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)

#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

## 2.1 利用`MultiIndex`构建器

pandas提供了专门的方法来构建`MultiIndex`：

```python
# list
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
#> MultiIndex(levels=[['a', 'b'], [1, 2]],
#>            labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

# tuple
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
#> MultiIndex(levels=[['a', 'b'], [1, 2]],
#>            labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

# Cartesian product
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
#> MultiIndex(levels=[['a', 'b'], [1, 2]],
#>            labels=[[0, 0, 1, 1], [0, 1, 0, 1]])

# 直接传入levels和labels
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
#> MultiIndex(levels=[['a', 'b'], [1, 2]],
#>            labels=[[0, 0, 1, 1], [0, 1, 0, 1]])        
```

## 2.2 给`MultiIndex` levels命名

有时候给`levels`命名可以使数据操作更方便(如果数据量较大，可以更好的理解每一列index表示什么意思)，可以通过传入`names`来实现：

```python
pop.index.names = ['state', 'year']
pop

#> state       year
#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

## 2.3 `MultiIndex`表示列

前面我们介绍的`MultiIndex`都是在行上的，实际上`DataFrame`的行和列是对称的，所以`MultiIndex`同样可以在列上使用。

```python
# 层次化的indices和columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# 生成一些数据
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# 创建DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data

#>      	subject Bob             Guido 	        Sue
#>      	type 	HR 	Temp 	HR 	Temp 	HR 	Temp
#> year 	visit 						
#> 2013 	1 	31.0 	38.7 	32.0 	36.7 	35.0 	37.2
#>              2 	44.0 	37.7 	50.0 	35.0 	29.0 	36.7
#> 2014 	1 	30.0 	37.4 	39.0 	37.8 	61.0 	36.9
#>              2 	47.0 	37.8 	48.0 	37.3 	51.0 	36.5
```

上面的表格就是一个基本的四维数据，分别表示subject, type, year, visit四个维度的数据。我们可以通过人名获得他的个人信息：

```python
health_data['Guido']

#> 	        type 	HR 	Temp
#> year 	visit 		
#> 2013 	1 	32.0 	36.7
#>              2 	50.0 	35.0
#> 2014 	1 	39.0 	37.8
#>              2 	48.0 	37.3
```

# 3. 索引和切片`MultiIndex`

## 3.1 多索引的`Series`

```python
pop

#> state       year
#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

```python
# 选取单个数据
pop['California', 2000]
#> 33871648

# partial indexing
pop['California']
#> year
#> 2000    33871648
#> 2010    37253956
#> dtype: int64

# Partial slicing
pop.loc['California':'New York']
#> state       year
#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> dtype: int64

pop[:, 2000]
#> state
#> California    33871648
#> New York      18976457
#> Texas         20851820
#> dtype: int64
```

之前介绍的indexing和selection方法同样适用：

```python
pop[pop > 22000000]
#> state       year
#> California  2000    33871648
#>             2010    37253956
#> Texas       2010    25145561
#> dtype: int64

pop[['California', 'Texas']]
#> state       year
#> California  2000    33871648
#>             2010    37253956
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

## 3.2 多索引的`DataFrame`

```python
health_data

#>      	subject Bob             Guido 	        Sue
#>      	type 	HR 	Temp 	HR 	Temp 	HR 	Temp
#> year 	visit 						
#> 2013 	1 	31.0 	38.7 	32.0 	36.7 	35.0 	37.2
#>              2 	44.0 	37.7 	50.0 	35.0 	29.0 	36.7
#> 2014 	1 	30.0 	37.4 	39.0 	37.8 	61.0 	36.9
#>              2 	47.0 	37.8 	48.0 	37.3 	51.0 	36.5
```

```python
health_data['Guido', 'HR']
#> year  visit
#> 2013  1        32.0
#>       2        50.0
#> 2014  1        39.0
#>       2        48.0
#> Name: (Guido, HR), dtype: float64

# iloc
health_data.iloc[:2, :2]
#> 	subject Bob
#> 	type   	HR 	Temp
#> year visit 		
#> 2013 1 	31.0 	38.7
#>      2 	44.0 	37.7

# loc
health_data.loc[:, ('Bob', 'HR')]
#> year  visit
#> 2013  1        31.0
#>       2        44.0
#> 2014  1        30.0
#>       2        47.0
#> Name: (Bob, HR), dtype: float64
```

注意，如果你想在index的元组中传入切片操作，pandas会报出语法错误：

```python
health_data.loc[(:, 1), (:, 'HR')]
#> SyntaxError: invalid syntax
```

要想实现这样的操作，可以使用`IndexSlice`：

```python
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]

#> subject      Bob Guido   Sue
#> type          HR    HR    HR
#> year visit                  
#> 2013 1      41.0  42.0  26.0
#> 2014 1      29.0  53.0  41.0
```


# 4. 重新排列`Multi-Indices`

对于多索引数据，了解如何有效的转换数据是关键。有很多操作可以保留数据的所有信息，但是在不同的计算目的下我们需要对其进行重新排列。我前面已经简单介绍了`stack()`和`unstack()`，除此之外还有很多其他操作。


## 4.1 已排序的(sorted)和未排序(unsorted)索引

前面我们介绍了`MultiIndex`的切片，实际上许多切片操作需要索引是排序好的(sorted)，否则进行切片的时候会报错：

```python
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data

#> char  int
#> a     1      0.799124
#>       2      0.968918
#> c     1      0.443728
#>       2      0.940282
#> b     1      0.287444
#>       2      0.448958
#> dtype: float64
```

我们来尝试对它进行去切片：

```python
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)

#> <class 'KeyError'>
#> 'Key length (1) was greater than MultiIndex lexsort depth (0)'
```

从错误提示我们可以明显看出来错误是由`MultiIndex`未排序导致的。pandas提供了一些用于排序的方法：`sort_index()`, `sortlevel()`。下面我们介绍一下最简单的方法`sort_index()`。

```python
data = data.sort_index()
data

#> char  int
#> a     1      0.799124
#>       2      0.968918
#> b     1      0.287444
#>       2      0.448958
#> c     1      0.443728
#>       2      0.940282
#> dtype: float64
```

我们再来对排序好的`data`进行切片试试：

```python
data['a':'b']

#> char  int
#> a     1      0.003001
#>       2      0.164974
#> b     1      0.001693
#>       2      0.526226
#> dtype: float64
```

## 4.2 融合(stack)与拆解(unstack)索引

正如我么前面见到的，我们可以把多索引的数据结构转化成二维数据。

```python
pop.unstack(level=0)
#> state 	California 	New York 	Texas
#> year 			
#> 2000 	33871648 	18976457 	20851820
#> 2010 	37253956 	19378102 	25145561

pop.unstack(level=1)
#> year 	2000 	2010
#> state 		
#> California 	33871648 	37253956
#> New York 	18976457 	19378102
#> Texas 	20851820 	25145561

pop.unstack().stack()
#> state       year
#> California  2000    33871648
#>             2010    37253956
#> New York    2000    18976457
#>             2010    19378102
#> Texas       2000    20851820
#>             2010    25145561
#> dtype: int64
```

## 4.3 索引设置与重置

我们还可以把索引标签(index label)转换成列。`reset_index`方法可以帮助我们实现这一功能。

```python
pop_flat = pop.reset_index(name='population')
pop_flat

#> 	state 	year 	population
#> 0 	California 	2000 	33871648
#> 1 	California 	2010 	37253956
#> 2 	New York 	2000 	18976457
#> 3 	New York 	2010 	19378102
#> 4 	Texas 	2000 	20851820
#> 5 	Texas 	2010 	25145561
```

# 5. `Multi-Indices`数据聚合(aggregation)

我们前面看到了pandas的一些聚合方法，比如`mean()`, `sum()`, `max()`。对于层次化索引的数据，这些方法可以通过传入`level`参数来控制小块数据的聚合。

```python
# 计算health_data每年两个测量数据的平均值
data_mean = health_data.mean(level='year')
data_mean

#> subject 	Bob 	       Guido    	Sue
#> type 	HR 	Temp 	HR 	Temp 	HR 	Temp
#> year 						
#> 2013 	37.5 	38.2 	41.0 	35.85 	32.0 	36.95
#> 2014 	38.5 	37.6 	43.5 	37.55 	56.0 	36.70
```

另外`axis`参数也是可用的

```python
data_mean.mean(axis=1, level='type')

#> type 	HR 	Temp
#> year 		
#> 2013 	36.833333 	37.000000
#> 2014 	46.000000 	37.283333
```


# 参考资料

1. [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product)
