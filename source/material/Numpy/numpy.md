# 一、Arrays简介

## 1.Numpy介绍
---
`Nunpy`是利用Python进行数据处理最基本，也是最强大的工具包。

如果你想从事数据分析或机器学习相关的工作，那么`numpy`是必须掌握的技能。因为其他数据分析包都是基于`numpy`的。比如`pandas`, `scikit-learn`等等。

那么`numpy`能为我们提供什么？

最核心的，`numpy`提供了十分优秀的`ndarray`对象，即n维数组。在`ndarray`中可以存储多个相同数据类型的数据，我们可以很方便的操作这些数据。

你可能会说：“我可以把数据存在python的列表中，我为什么还需要numpy数组呢？”

实际上，numpy数组相比list有很大优势，为了了解这些优势，我们先来看看怎么创建一个数组。

## 2.如何创建numpy数组？
---
创建数组的方法很多，其中最重要的是把一个列表传入到`np.array`方程中。
```python
#　从列表创建一个一维数组
import numpy as np

list1 = [0, 1, 2, 3, 4]
arr1d = np.array(list1)

#　打印数组和类型
print(type(arr1d))
print(arr1d)

#> class 'numpy.ndarray'
#> array([0, 1, 2, 3, 4])
```
- `array`和`list`最重要的差别就是`array`可以用来处理向量操作，而`list`不可以。如下例：

```python
list1 + 2
# 报错

arr1d + 2
#> array([2, 3, 4, 5, 6])
```
- 另一个不同点是，一旦`array`被创建了它的size就不能改变了，如果想改变数组size就只能创建一个新的数组。而`list`的size是可以随意改变的。

- array还可以通过`dtype`参数来指定数据类型。常用的有：`float`, `int`, `bool`, `str`和`object`。为了控制内存还可以选择`float64`, `float32`, `int8`, `int16`或者`int32`。

- `array`还可以通过`astype`方法来改变数据类型。

```python
# 从嵌套列表创建一个2d array,并指定数据类型'float'
list2 = [[0,1,2], [3,4,5], [6,7,8]]
arr2d = np.array(list2，　dtype='float')
arr2d

#> array([[0., 1., 2.],
#>        [3., 4., 5..],
#>        [6., 7., 8.]])

# 转换成int类型
arr2d.astype('int')

#> array([[0, 1, 2],
#>        [3, 4, 5],
#>        [6, 7, 8]])
```

- `array`中的元素必须具有相同的数据类型，而`list`中的元素可以不同。然而，如果你想令字符串和数字同时在一个`array`中，可以指定`dtype`为`object`。

```python
arr1d_obj = np.array([1, 'a'], dtype='object')
arr1d_obj

#> array([1, 'a'], dtype=object)
```
- 最后，你还可以通过`tolist`方法将`array`转换成`list`。

```python
arr1d_obj.tolist()

#> [1, 'a']
```
**总结**
> 1. array支持向量操作，list不支持
> 2. array不可以改变size, list可以
> 3. 每个array有且仅有一个数据类型
> 4. 等价的array比list占用更少的内存

## 3.如何查看array的size和shape?
---
- array维度(ndim)
- array形状(shape)
- array数据类型
- array元素个数(size)

```python
# 创建一个３×４的array，数据类型是‘float’
list2 = [[1, 2, 3, 4],[3, 4, 5, 6], [5, 6, 7, 8]]
arr2 = np.array(list2, dtype='float')
arr2

#> array([[ 1.,  2.,  3.,  4.],
#>        [ 3.,  4.,  5.,  6.],
#>        [ 5.,  6.,  7.,  8.]])

# shape
print('Shape: ', arr2.shape)

# dtype
print('Datatype: ', arr2.dtype)

# size
print('Size: ', arr2.size)

# ndim
print('Num Dimensions: ', arr2.ndim)

#> Shape:  (3, 4)
#> Datatype:  float64
#> Size:  12
#> Num Dimensions:  2
```

## 4.如何抽取array中指定数据？
---

我们可以通过索引来提取指定的数据，索引从０开始计数。array的索引类似于list但又有所不同：

```python
# 提取第一个元素
arr2[0]
# array([ 1.,  2., 3., 4.])

list2[0]
#[1, 2, 3, 4]

# 提取前两行和前两列
arr2[:2, :2]

#> array([[ 1.,  2.],
#>        [ 3.,  4.]])

list2[:2, :2]
# 报错
```

另外，array支持布尔索引：

```python
b = arr2 > 4
b

#> array([[False, False, False, False],
#>        [False, False,  True,  True],
#>        [ True,  True,  True,  True]], dtype=bool)

arr2[b]

#> array([ 5.,  6.,  5.,  6.,  7.,  8.])
```

### 4.1 如何翻转行和整个array？

```python
# 只翻转行
arr2[::-1, ]

#> array([[ 5.,  6.,  7.,  8.],
#>        [ 3.,  4.,  5.,  6.],
#>        [ 1.,  2.,  3.,  4.]])

# 翻转整个array
arr2[::-1, ::-1]

#> array([[ 8.,  7.,  6.,  5.],
#>        [ 6.,  5.,  4.,  3.],
#>        [ 4.,  3.,  2.,  1.]])
```

### 4.2 如何表示缺失值和无限值？

缺失值用`np.nan`对象表示，无限值用`np.inf`对象表示。

```python
# 插入nan和inf
arr2[1,1] = np.nan  # not a number
arr2[1,2] = np.inf  # infinite
arr2

#> array([[  1.,   2.,   3.,   4.],
#>        [  3.,  nan,  inf,   6.],
#>        [  5.,   6.,   7.,   8.]])

# 不用arr2 == np.nan来替换nan和inf为-１
missing_bool = np.isnan(arr2) | np.isinf(arr2)
arr2[missing_bool] = -1  
arr2

#> array([[ 1.,  2.,  3.,  4.],
#>        [ 3., -1., -1.,  6.],
#>        [ 5.,  6.,  7.,  8.]])
```

### 4.3 如何计算均值，最大值，最小值？

计算整个array中的均值，最大最小值：
```python
# mean, max and min
print("Mean value is: ", arr2.mean())
print("Max value is: ", arr2.max())
print("Min value is: ", arr2.min())

#> Mean value is:  3.58333333333
#> Max value is:  8.0
#> Min value is:  -1.0
```

计算某一个轴上的最大最小值用`np.amax`或`np.amin`来代替：
```python
print("Column minimum: ", np.amin(arr2, axis=0))
print("Row minimum: ", np.amin(arr2, axis=1))

#> Column wise minimum:  [ 1. -1. -1.  4.]
#> Row wise minimum:  [ 1. -1.  5.]
```

## 5. 如何从一个已存在的array创建新的array？
---

如果你只是将一个数组中一部分分配给另一个数组，实际上这个新数组在内存中指向父数组的。这就意味着对新数组的任何改变都会影响父数组。

所以，为了避免影响父数组，你需要用`copy()`来拷贝数组。

```python
arr2a = arr2[:2,:2]  
arr2a[:1, :1] = 100  # 100 will reflect in arr2
arr2

#> array([[ 100.,    2.,    3.,    4.],
#>        [   3.,   -1.,   -1.,    6.],
#>        [   5.,    6.,    7.,    8.]])
```

```python
arr2b = arr2[:2, :2].copy()
arr2b[:1, :1] = 101  # 101 will not reflect in arr2
arr2

#> array([[ 100.,    2.,    3.,    4.],
#>        [   3.,   -1.,   -1.,    6.],
#>        [   5.,    6.,    7.,    8.]])
```

## 6. Reshaping和Flattening多维数组
---
- Reshaping是数组元素不变的情况下改变数组的形状
- Flattening是将多维数组转换成１维

```python
arr2.reshape(4, 3)

#> array([[ 100.,    2.,    3.],
#>        [   4.,    3.,   -1.],
#>        [  -1.,    6.,    5.],
#>        [   6.,    7.,    8.]])

arr2.flatten()

#>array([100.,   2.,   3.,   4.,   3.,  -1.,  -1.,   6.,   5.,   6.,   7., 8.])
```

### 6.1 flatten()和ravel()有何不同？

有两种方法可以实现flattening：`flatten()`和`ravel()`。那么这两种方法有何不同呢？

`ravel()`是对父数组的一个引用，对数组中任何元素的改变都会影响父数组；而`flatten()`会创建一个copy，不会影响父数组。

```python
# 改变flattened array不影响父数组
b1 = arr2.flatten()  
b1[0] = 10  # changing b1 does not affect arr2
arr2

#> array([[ 100.,    2.,    3.,    4.],
#>        [   3.,   -1.,   -1.,    6.],
#>        [   5.,    6.,    7.,    8.]])

# 改变raveled array也会改变父数组
b2 = arr2.ravel()  
b2[0] = 10  # changing b2 changes arr2 also
arr2

#> array([[ 10.,    2.,    3.,    4.],
#>        [   3.,   -1.,   -1.,    6.],
#>        [   5.,    6.,    7.,    8.]])
```

## 7. 如何创建序列，重复和任意数字？
---

`np.arange()`函数用户自定义创建一个序列作为`ndarray`。

```python
# 下限默认为０
print(np.arange(5))  

# 0 - 9
print(np.arange(0, 10))  

# 0 - 9 步长为2
print(np.arange(0, 10, 2))  

# 10 - 1, 降序
print(np.arange(10, 0, -1))

#> [0 1 2 3 4]
#> [0 1 2 3 4 5 6 7 8 9]
#> [0 2 4 6 8]
#> [10  9  8  7  6  5  4  3  2  1]
```

当我们需要手动计算步长的时候，我们可以用`np.linspace()`来替代。

```python
# 1 - 50, 10个等间距数
np.linspace(start=1, stop=50, num=10)

#> array([ 1.        ,  6.44444444, 11.88888889, 17.33333333, 22.77777778, 28.22222222, 33.66666667, 39.11111111, 44.55555556, 50.        ])

# 强制转化为int类型
np.linspace(start=1, stop=50, num=10, dtype=int)

#> array([ 1,  6, 11, 17, 22, 28, 33, 39, 44, 50])
```
与`np.linspace()`类似，还有一个`np.logspace()`函数。`np.logspace()`函数中start, stop参数分别表示$$base^{start}$$和$$base^{stop}$$，默认base为10。
```python
# 小数点保留两位
np.set_printoptions(precision=2)  

# 10^1 - 10^50
np.logspace(start=1, stop=50, num=10, base=10)

#> array([  1.00e+01,   2.78e+06,   7.74e+11,   2.15e+17,   5.99e+22,
#>          1.67e+28,   4.64e+33,   1.29e+39,   3.59e+44,   1.00e+50])
```

`np.zeros()`和`np.ones()`函数可以创建指定shape的全0或全1的array。

```python
np.zeros([2,2])
#> array([[ 0.,  0.],
#>        [ 0.,  0.]])

np.ones([2,2])
#> array([[ 1.,  1.],
#>        [ 1.,  1.]])
```

### 7.1 如何创建重复序列？

`np.tile()`会重复整个列表或数组n次，`np.repeat()`会将每个元素重复n次。

```python
a = [1,2,3]

# Repeat whole of 'a' two times
print('Tile:   ', np.tile(a, 2))

# Repeat each element of 'a' two times
print('Repeat: ', np.repeat(a, 2))

#> Tile:    [1 2 3 1 2 3]
#> Repeat:  [1 1 2 2 3 3]
```

### 7.2 如何生成随机数？

`random`模块提供了一系列方程用于生成随机数和给定shape的统计分布。

```python
# [0,1)之间， shape 2,2随机分布
print(np.random.rand(2,2))

# mean=0, variance=1，shape 2,2正态分布
print(np.random.randn(2,2))

# [0, 10)之间，shape 2,2随机整数分布
print(np.random.randint(0, 10, size=[2,2]))

# [0,1)之间随机数
print(np.random.random())

# [0,1), shape 2,2随机分布
print(np.random.random(size=[2,2]))

# 从给定列表中等概率随机挑选10个元素
print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10))  

# 从给定列表中按照'p'概率分布随机挑选10个元素
print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10, p=[0.3, .1, 0.1, 0.4, 0.1]))

#> [[ 0.84  0.7 ]
#>  [ 0.52  0.8 ]]

#> [[-0.06 -1.55]
#>  [ 0.47 -0.04]]

#> [[4 0]
#>  [8 7]]

#> 0.08737272424956832

#> [[ 0.45  0.78]
#>  [ 0.03  0.74]]

#> ['i' 'a' 'e' 'e' 'a' 'u' 'o' 'e' 'i' 'u']
#> ['o' 'a' 'e' 'a' 'a' 'o' 'o' 'o' 'a' 'o']
```

现在，你每次运行上面的方程得到的数字或分布都会不一样。

如果你想每次运行结果都相同你需要设置一个随机状态的“种子”，这个种子可以是任意数字，唯一的要求是每次运行的时候，“种子”的值必须相同。

```python
# 创建随机状态
rn = np.random.RandomState(100)

# [0,1)，　shape 2,2
print(rn.rand(2,2))

#> [[ 0.54  0.28]
#>  [ 0.42  0.84]]
```

```python
# 设置随机种子
np.random.seed(100)

# [0,1)， 2,2
print(np.random.rand(2,2))

#> [[ 0.54  0.28]
#>  [ 0.42  0.84]]
```

### 7.3 如何获得唯一值并计数？

`np.unique`方法可以用来获取唯一值，如果你想得到这个值出现了几次，可以把`return_counts`参数设置为`True`。

```python
np.random.seed(100)
arr_rand = np.random.randint(0, 10, size=10)
print(arr_rand)

#> [8 8 3 7 7 0 4 2 5 2]

# 获得唯一值并计数
uniqs, counts = np.unique(arr_rand, return_counts=True)
print("Unique items : ", uniqs)
print("Counts       : ", counts)

#> Unique items :  [0 2 3 4 5 7 8]
#> Counts       :  [1 2 1 1 1 2 2]
```

# 二、用于数据分析的重要方程

## 1. 如何得到满足给定条件的数据的索引位置？
---

之前我们提到了，如何通过布尔索引获得满足条件的数据。那么，我们如何得到满足条件的数据所在的位置呢？

`np.where()`可以帮助我们实现这一想法。

```python
# 创建一个array
import numpy as np
arr_rand = np.array([8, 8, 3, 7, 7, 0, 4, 2, 5, 2])
print("Array: ", arr_rand)

# 大于5的数据位置
index_gt5 = np.where(arr_rand > 5)
print("Positions where value > 5: ", index_gt5)

#> Array:  [8 8 3 7 7 0 4 2 5 2]
#> Positions where value > 5:  (array([0, 1, 3, 4]),)
```

当你获得了这些索引的时候，就可以通过`take`方法来获得数据值。

```python
arr_rand.take(index_gt5)

#> array([[8, 8, 7, 7]])
```

`np.where()`可以额外接收２个参数——`x`和`y`，满足条件的数据设置成'x'，否则为'y'。

```python
# 把大于５的值设为‘gt5’，否则为'le5'
np.where(arr_rand > 5, 'gt5', 'le5')

#> array(['gt5', 'gt5', 'le5', 'gt5', 'gt5', 'le5', 'le5', 'le5', 'le5', 'le5'], dtype='<U3')
```

让我们来找到最大值，最小值的位置

```python
# 最大值位置
print('Position of max value: ', np.argmax(arr_rand))  

# 最小值位置
print('Position of min value: ', np.argmin(arr_rand))  

#> Position of max value:  0
#> Position of min value:  5
```

## 2. 如何导入和导出数据(*.csv)
---
一个标准的导入数据的方法是使用`np.genfromtxt`函数。该函数可以从URLs中导入数据，处理缺失值，多分隔符，处理列数不一致等情况。

另一个并不这么通用的方法是使用`np.loadtxt`函数，该函数假设数据没有缺失值。

```python
# 从url链接导入数据
path = 'https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv'
data = np.genfromtxt(path, delimiter=',', skip_header=1, filling_values=-999, dtype='float')　# 可以用filling_values参数将缺失值替换成任意值
data[:3]  # 看前３行数据

#> array([[   18. ,     8. ,   307. ,   130. ,  3504. ,    12. ,    70. ,　 1. ,  -999. ],
#>        [   15. ,     8. ,   350. ,   165. ,  3693. ,    11.5,    70. , 　1. ,  -999. ],
#>        [   18. ,     8. ,   318. ,   150. ,  3436. ,    11. ,    70. ,　 1. ,  -999. ]])
```

### 2.1 如何处理既有数字又有文字的数据集？

如果必须保留文字列的话，可以将`dtype`参数设置成`None`或`object`。

```python
# data2 = np.genfromtxt(path, delimiter=',', skip_header=1, dtype='object')
data2 = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=None)
data2[:3]

#> array([( 18., 8,  307., 130, 3504,  12. , 70, 1, b'"chevrolet chevelle malibu"'),
#>        ( 15., 8,  350., 165, 3693,  11.5, 70, 1, b'"buick skylark 320"'),
#>        ( 18., 8,  318., 150, 3436,  11. , 70, 1, b'"plymouth satellite"')],
#>       dtype=[('f0', '<f8'), ('f1', '<i8'), ('f2', '<f8'), ('f3', '<i8'), ('f4', '<i8'), ('f5', '<f8'), ('f6', '<i8'), ('f7', '<i8'), ('f8', 'S38')])
```

最后，使用`np.savetxt`函数导出数据

```python
np.savetxt("out.csv", data, delimiter=",")
```

## 3. 如何保存和载入numpy对象？

有时候我们希望保存大量的经过变换的numpy arrays到硬盘，然后直接把他们载入到控制台，而不需要重新跑一边程序。

Numpy提供了两种数据格式：`.npy`和`npz`。

如果你想存储单个`ndarray`对象，可以用`np.save`函数将数据存成`.npy`格式数据文件，然后用`np.load`函数载入数据。

如果你想保存多余１个`ndarray`对象，就要用`np.savez`函数把数据存成`.npz`格式文件。

```python
# 保存单个numpy array object为.npy
np.save('myarray.npy', arr2d)  

# 保存多个numy arrays为.npz
np.savez('array.npz', arr2d_f, arr2d_b)

# 载入.npy
a = np.load('myarray.npy')

# 载入.npz
b = np.load('array.npz')
```

## 4. 如何拼接两个numpy arrays
---
有三种方法来拼接两个或两个以上的numpy arrays：

- `np.concatenate`通过设置`axis`参数来控制拼接方向。
- `np.vstack`和`np.hstack`
- `np.r_`和`np.c_`

以上三种方法可以得到相同的输出。

与其他两种方法不同的是，`np.r_`和`np.c_`使用方括号来进行拼接。

```python
# 创建两个数组
a = np.zeros([4, 4])
b = np.ones([4, 4])
print(a)
print(b)

#> [[ 0.  0.  0.  0.]
#>  [ 0.  0.  0.  0.]
#>  [ 0.  0.  0.  0.]
#>  [ 0.  0.  0.  0.]]

#> [[ 1.  1.  1.  1.]
#>  [ 1.  1.  1.  1.]
#>  [ 1.  1.  1.  1.]
#>  [ 1.  1.  1.  1.]]

#沿着纵轴拼接
np.concatenate([a, b], axis=0)  
np.vstack([a,b])  
np.r_[a,b]  

#> array([[ 0.,  0.,  0.,  0.],
#>        [ 0.,  0.,  0.,  0.],
#>        [ 0.,  0.,  0.,  0.],
#>        [ 0.,  0.,  0.,  0.],
#>        [ 1.,  1.,  1.,  1.],
#>        [ 1.,  1.,  1.,  1.],
#>        [ 1.,  1.,  1.,  1.],
#>        [ 1.,  1.,  1.,  1.]])

# 沿着横轴拼接
np.concatenate([a, b], axis=1)
np.hstack([a,b])  
np.c_[a,b]

#> array([[ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
#>        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
#>        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
#>        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.]])
```

## 5. 如何基于一列或多列数据对numpy array进行排序?
---

让我们试着对一个2d数组，根据第一列进行排序。

```python
arr = np.random.randint(1,6, size=[8, 4])
arr

#> array([[3, 3, 2, 1],
#>        [1, 5, 4, 5],
#>        [3, 1, 4, 2],
#>        [3, 4, 5, 5],
#>        [2, 4, 5, 5],
#>        [4, 4, 4, 2],
#>        [2, 4, 1, 3],
#>        [2, 2, 4, 3]])
```

我们有一个８行４列的随机数组。

如果你想用`np.sort`函数`axis=0`来排序的话，整个array每一列都会单独进行降序排序，影响行数据的完整性。

```python
np.sort(arr, axis=0)

#> array([[1, 1, 1, 1],
#>        [2, 2, 2, 2],
#>        [2, 3, 4, 2],
#>        [2, 4, 4, 3],
#>        [3, 4, 4, 3],
#>        [3, 4, 4, 5],
#>        [3, 4, 5, 5],
#>        [4, 5, 5, 5]])
```
我们不想打乱行数据，我们可以使用`np.argsort`方法间接进行排序。

### 5.1 如何使用`np.argsort`基于第一列数据对array进行排序？

让我们先搞清楚`np.argsort`是在做什么。

`np.argsort`会返回一个排序后的数组的索引。

```python
x = np.array([1, 10, 5, 2, 8, 9])
sort_index = np.argsort(x)
print(sort_index)

#> [0 3 2 4 5 1]

x[sort_index]

#> array([ 1,  2,  5,  8,  9, 10])
```

现在，为了对原始的`array`进行排序，我打算对第一列进行一次argsort排序，然后用返回的结果来排序`array`。

```python
# 对第一列进行argsort排序
sorted_index_1stcol = arr[:, 0].argsort()

# 对arr进行排序
arr[sorted_index_1stcol]

#> array([[1, 5, 4, 5],
#>        [2, 4, 5, 5],
#>        [2, 4, 1, 3],
#>        [2, 2, 4, 3],
#>        [3, 3, 2, 1],
#>        [3, 1, 4, 2],
#>        [3, 4, 5, 5],
#>        [4, 4, 4, 2]])
```

为了可以降序排序，可以用如下方法：

```python
arr[sorted_index_1stcol[::-1]]

#> array([[4, 4, 4, 2],
#>        [3, 4, 5, 5],
#>        [3, 1, 4, 2],
#>        [3, 3, 2, 1],
#>        [2, 2, 4, 3],
#>        [2, 4, 1, 3],
#>        [2, 4, 5, 5],
#>        [1, 5, 4, 5]])
```

### 5.2 如何根据2列或更多列进行排序？

你可以使用`np.lexsort`函数，传入一个元组（tuple）来实现。

```python
# 先根据第２列排序，然后根据第１列进行排序
lexsorted_index = np.lexsort((arr[:, 1], arr[:, 0]))
arr[lexsorted_index]

#> array([[1, 5, 4, 5],
#>        [2, 2, 4, 3],
#>        [2, 4, 5, 5],
#>        [2, 4, 1, 3],
#>        [3, 1, 4, 2],
#>        [3, 3, 2, 1],
#>        [3, 4, 5, 5],
#>        [4, 4, 4, 2]])
```

## 6. 日期时间序列
---
Numpy通过`np.datetime64`对象实现了日期，精度可达纳秒。你可以通过标准的YYYY-MM-DD格式字符串创建一个日期。

```python
date64 = np.datetime64('2018-02-04 23:10:10')
date64

#> numpy.datetime64('2018-02-04T23:10:10')
```
当然，你还可以传入小时，分钟，秒等等。

现在让我们从`date64`中删除时间成分。

```python
dt64 = np.datetime64(date64, 'D')
dt64

#> numpy.datetime64('2018-02-04')
```

默认如果你添加一个数字，会增加天数。但是如果你想改变其他时间，可以通过`timedelta`对象。

```python
tenminutes = np.timedelta64(10, 'm')  # 10 minutes
tenseconds = np.timedelta64(10, 's')  # 10 seconds
tennanoseconds = np.timedelta64(10, 'ns')  # 10 nanoseconds

print('Add 10 days: ', dt64 + 10)
print('Add 10 minutes: ', dt64 + tenminutes)
print('Add 10 seconds: ', dt64 + tenseconds)
print('Add 10 nanoseconds: ', dt64 + tennanoseconds)

#> Add 10 days:  2018-02-14
#> Add 10 minutes:  2018-02-04T00:10
#> Add 10 seconds:  2018-02-04T00:00:10
#> Add 10 nanoseconds:  2018-02-04T00:00:00.000000010
```

现在让我们把`dt64`转换成字符串

```python
np.datetime_as_string(dt64)

#> '2018-02-04'
```

有时候我们需要过滤出工作日，可以用`np.is_busday()`。

```python
print('Date: ', dt64)
print("Is it a business day?: ", np.is_busday(dt64))  
print("Add 2 business days, rolling forward to nearest biz day: ", np.busday_offset(dt64, 2, roll='forward'))  
print("Add 2 business days, rolling backward to nearest biz day: ", np.busday_offset(dt64, 2, roll='backward'))  

#> Date:  2018-02-04
#> Is it a business day?:  False
#> Add 2 business days, rolling forward to nearest biz day:  2018-02-07
#> Add 2 business days, rolling backward to nearest biz day:  2018-02-06
```

### 6.1 如何创建日期序列？

可以通过`np.arange`来完成

```python
# 创建日期序列
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-10'))
print(dates)

# 检查是否是工作日
np.is_busday(dates)

#> ['2018-02-01' '2018-02-02' '2018-02-03' '2018-02-04' '2018-02-05' '2018-02-06' '2018-02-07' '2018-02-08' '2018-02-09']
#>　array([ True,  True, False, False,  True,  True,  True,  True,  True], dtype=bool)
```

### 6.2 如何将`numpy.datetime64`转化成`datetime.datetime`对象？

```python
import datetime
dt = dt64.tolist()
dt

#> datetime.date(2018, 2, 4)
```
一旦转化成`datetime.datetime`，你就有更多的工具来提取年，月，日，星期等信息。

```python
print('Year: ', dt.year)  
print('Day of month: ', dt.day)
print('Month of year: ', dt.month)  
print('Day of Week: ', dt.weekday())

#> Year:  2018
#> Day of month:  4
#> Month of year:  2
#> Day of Week:  6  #Sunday
```

## 7. Numpy中高级函数
---
### 7.1 `vectorize`－使标量函数作用在向量上
在`vectorize()`的帮助下，你可以使一个标量函数作用于数组上。

让我们看一个简单的例子。

`foo`函数接收一个数字，如果是奇数就进行平方，否则就除以２。

当你作用与一个标量上时，可以工作，一旦传入一个数组就会报错。

使用`vectorize()`，你可以使函数完美工作。

```python
# 定义一个标量函数
def foo(x):
    if x % 2 == 1:
        return x**2
    else:
        return x/2

# 传入标量
print('x = 10 returns ', foo(10))
print('x = 11 returns ', foo(11))

#> x = 10 returns  5.0
#> x = 11 returns  121

# 传入数组
print('x = [10, 11, 12] returns ', foo([10, 11, 12]))  

#> 报错
```

```python
# 向量化foo()
foo_v = np.vectorize(foo, otypes=[float])

print('x = [10, 11, 12] returns ', foo_v([10, 11, 12]))
print('x = [[10, 11, 12], [1, 2, 3]] returns ', foo_v([[10, 11, 12], [1, 2, 3]]))


#> x = [10, 11, 12] returns  [   5.  121.    6.]
#> x = [[10, 11, 12], [1, 2, 3]] returns  [[   5.  121.    6.]
#> [   1.    1.    9.]]
```

`otypes`参数可以指定输出数据类型，使向量化的函数运行更快。

### 7.2 `apply_along_axis` ——　在列或者行上使用函数

首先创建一个二维数组

```python
np.random.seed(100)
arr_x = np.random.randint(1,10,size=[4,10])
arr_x

#> array([[9, 9, 4, 8, 8, 1, 5, 3, 6, 3],
#>        [3, 3, 2, 1, 9, 5, 1, 7, 3, 5],
#>        [2, 6, 4, 5, 5, 4, 8, 2, 2, 8],
#>        [8, 1, 3, 4, 3, 6, 9, 2, 1, 8]])
```

如何找到每一行或每一列的最大值和最小值？

通常的方法是用循环。听起来还不错，但是如果数据量很大，并且每一行或者列还要进行复杂的运算，那么这种办法非常笨重。

你么可以非常优雅的使用`numpy.apply_along_axis`。

参数：

１．需要用到的函数
２．作用轴
３．数组

```python
# 定义函数
def max_minus_min(x):
    return np.max(x) - np.min(x)

# 作用在行上
print('Row wise: ', np.apply_along_axis(max_minus_min, 1, arr=arr_x))

# 作用在列上
print('Column wise: ', np.apply_along_axis(max_minus_min, 0, arr=arr_x))

#> Row wise:  [8 8 6 8]
#> Column wise:  [7 8 2 7 6 5 8 5 5 5]
```

### 7.3 `searchsorted` ——　找到应该插入的位置保持数组仍然数排序好的

```python
x = np.arange(10)
print('Where should 5 be inserted?: ', np.searchsorted(x, 5))
print('Where should 5 be inserted (right)?: ', np.searchsorted(x, 5, side='right'))

#> Where should 5 be inserted?:  5
#> Where should 5 be inserted (right)?:  6
```

你可以使用`searchsorted`来代替`np.choice`对数据进行采样，速度会快很多。

```python
# 根据预定义的概率分布，随机挑选一个元素
lst = range(10000)  # the list
probs = np.random.random(10000); probs /= probs.sum()  # probabilities

%timeit lst[np.searchsorted(probs.cumsum(), np.random.random())]
%timeit np.random.choice(lst, p=probs)


#> 10000 loops, best of 3: 33 µs per loop
#> 1000 loops, best of 3: 1.32 ms per loop
```

### 7.4 如何给`numpy array`增加一个轴？

有时候我们希望在不增加数据量的同时，给数据增加一个维度。

`np.newaxis`可以帮助我们将一个低纬度数据转化成高纬度。

```python
# 创建一个1darray
x = np.arange(5)
print('Original array: ', x)

# 引入一个新列
x_col = x[:, np.newaxis]
print('x_col shape: ', x_col.shape)
print(x_col)

# 引入一个新行
x_row = x[np.newaxis, :]
print('x_row shape: ', x_row.shape)
print(x_row)

#> Original array:  [0 1 2 3 4]
#> x_col shape:  (5, 1)
#> [[0]
#>  [1]
#>  [2]
#>  [3]
#>  [4]]
#> x_row shape:  (1, 5)
#> [[0 1 2 3 4]]
```

### 7.5 更多有用的函数

**Digitize**

返回索引所在位置属于哪个bin。

```python
x = np.arange(10)
bins = np.array([0, 3, 6, 9])

np.digitize(x, bins)

#> array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
```

**Clip**

使数据处于给定范围内，低于下限的全部设为最低值，高于上限的全部设为最高值。

```python
np.clip(x, 3, 8)

#> array([3, 3, 3, 3, 4, 5, 6, 7, 8, 8])
```

**Histogram and Bincount**

`histogram()`和`bincount()`都可以给出出现的次数，但是二者有明显的区别。

`histogram()`给出bins的计数，`bincount()`给出整个array最大最小值范围内所有元素出现的次数，包括没有出现的。

```python
# Bincount
x = np.array([1,1,2,2,2,4,4,5,6,6,6]) # doesn't need to be sorted
np.bincount(x)

#> array([0, 2, 3, 0, 2, 1, 3])
# 0出现0次，1出现2次，２出现３次...

# Histogram
counts, bins = np.histogram(x, [0, 2, 4, 6, 8])
print('Counts: ', counts)
print('Bins: ', bins)

#> Counts:  [2 3 3 3]
#> Bins:  [0 2 4 6 8]
```

