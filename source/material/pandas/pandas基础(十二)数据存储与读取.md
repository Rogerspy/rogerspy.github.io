<h1>Pandas基础教程（十二）数据存储与读取</h1>

[TOC]

本节是pandas系列学习笔记的最后一节，必定有很多内容没有覆盖到，只有在实际应用过程中遇到问题多多查看帮助文档，多交流才能更进一步。

```python
import numpy as np
import pandas as pd
```

# 1. 读取文本格式的数据

pandas提供了一系列将表格数据读取为`DataFrame`对象的函数，主要包括：

|函数|说明|
|-|-|
|<code>read_csv</code>|从文件、url等对象中加载带分隔符的数据，默认为逗号（`,`）|
|<code>read_table</code>|从文件、url等对象中加载带分隔符的数据，默认为制表符（`\t`）|
|<code>read_fwf</code>|读取定宽列格式数据，也就是说没有分隔符|
|<code>read_clipboard</code>|从剪切板读取数据|

这些函数有一个重要的功能就是类型推断，也就是说，不需要指定数据类型，函数可以自己推断数据类型。以最常用的`read_csv()`为例，假设我们现在有一个名为“ex1.csv”的数据文件, 内容如下：

```
a,b,c,d,message
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo
```

```python
df = pd.read_csv('ex1.csv')
df

#>    a   b   c   d message
#> 0  1   2   3   4   hello
#> 1  5   6   7   8   world
#> 2  9  10  11  12     foo
```

pandas默认将第一行的数据作为列名，但有时候我们的数据不包含列名，如果使用默认使用第一行作为列名可能造成一些问题，对于这个问题我们有两种解决方法：

```python
# 使用header参数来指定不需要列名
pd.read_csv('ex1.csv', header=None)
#> 0   a   b   c   d message
#> 1  1   2   3   4   hello
#> 2  5   6   7   8   world
#> 3  9  10  11  12     foo

# 自己定义列名
pd.read_csv('ex1.csv', names=['c1', 'c2', 'c3', 'c4', 'c5'])
#>   c1  c2  c3  c4       c5
#> 0  a   b   c   d  message
#> 1  1   2   3   4    hello
#> 2  5   6   7   8    world
#> 3  9  10  11  12      foo
```

假设我们希望‘message’作为索引，我们可以使用`index_col`参数来指定：

```python
pd.read_csv('ex1.csv', index_col='message')

#>          a   b   c   d
#> message               
#> hello    1   2   3   4
#> world    5   6   7   8
#> foo      9  10  11  12
```

如果我们想将多个列做成一个层次化索引，只需将相应的列传入列表然后传给`index_col`即可。假设我们有另外一个数据文件'ex2.csv', 内容如下：
```
key1,key2,value1,velue2
one,a,1,2
one,b,3,4
one,c,5,6
one,d,7,8
two,a,9,10
two,b,11,12
```

```python
pd.read_csv('ex2.csv', index_col=['key1','key2'])

#>            value1  velue2
#> key1 key2                
#> one  a          1       2
#>      b          3       4
#>      c          5       6
#>      d          7       8
#> two  a          9      10
#>      b         11      12
```

有些表格的分隔符可能不固定，这时候我们可以使用`pd.read_table`，并用正则表达式来表示分隔符。假设我们有一个数据文件'ex3.csv'，　内容如下：

```
a   b   c   d
0.9041,0.0227   0.9230,0.4260
0.5736,0.0369     0.2902,0.7425
0.6216,0.4500        0.3512,0.1131
0.1759,0.7114    0.0825,0.3438
```

有逗号，有空格，有制表符作为分隔符。

```python
pd.read_table('ex3.csv', sep='[\s,]+')

#>         a       b       c       d
#> 0  0.9041  0.0227  0.9230  0.4260
#> 1  0.5736  0.0369  0.2902  0.7425
#> 2  0.6216  0.4500  0.3512  0.1131
#> 3  0.1759  0.7114  0.0825  0.3438
```

另外`skiprows`参数还可以帮助我们跳过一些行。假设我们有一二数据文件'ex4.csv'，内容如下：

```
# sadsfsdfdsfsdfda
#xddgdfsfsdsdv
#fddffbgbfvvv
@dffdfdgfdgdf
a,b,c,d,message
1,2,3,4,hello
#gghgf
5,6,7,8,world
9,10,11,12,foo
```

```python
pd.read_csv('ex4.csv', skiprows=[0,1,2,3,6])

#>    a   b   c   d message
#> 0  1   2   3   4   hello
#> 1  5   6   7   8   world
#> 2  9  10  11  12     foo
```

如果我们并不像读取整个文件，而是只读取其中的若干行，可以使用`nrows`参数，如果要逐块读取数据可以通过`chunksize`参数。

# 2. 将数据写到文件

数据可以输出为分隔符格式的文本，比如csv:

```python
data.to_csv('ex6.csv')
```

默认分隔符为`,`，可以使用`sep`来指定分隔符。

```python
data.to_csv('ex6.csv', sep='|')
```

缺失值写入到文件之后会以空字符串代替。

默认情况下，pandas也会将行列名称写进文件，如果我们不希望如此，可以使用`index＝False`和`header=False`来禁用。

```python
data.to_csv('ex6.csv', index=False, header=False)
```

还可以只写一部分列数据：

```python
data.to_csv('ex6.csv',cols=['a', 'b', 'c'])
```

以上表示只写'a', 'b', 'c'三列的数据。


关于pandas数据文件存取的内容我们就简单学习到这里，更多操作可以查看帮助文档，在实践中学习才能走的更远。
