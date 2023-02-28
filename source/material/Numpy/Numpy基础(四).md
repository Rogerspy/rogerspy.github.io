<h1>Numpy基础教程（四）用于数组的文件输入输出</h1>

[TOC]

# 1. 将数组以二进制格式保存到磁盘

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

# 2. 存取文本文件(*.csv)

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

`np.savetxt`将数组写入到某种分隔符分开的文本文件中。
