<h1>Numpy基础教程（七）内存映射数组</h1>

[TOC]

# 1. numpy.memmap

当我们需要存取一个很大的数据文件中的一小部分时，如果把整个文件读入内存会很耗费资源。使用内存映射数组可以帮助我们快速解决问题。

`numpy.memmap(filename, dtype, mode, offset, shape, order)`:

> `filename`:需要读取的文件。
> `dtype`: 数据类型
> `mode`: {'r+', 'r', 'w+', 'c'}, 分别表示“打开已存在的文件，只读”，“打开已存在的文件，读写”，“创建或覆盖已存在的文件，读写”，“拷贝可写，影响内存中的数据，但不影响硬盘上的数据”
> `offset`:文件中存储数据的开始位置，以字节为单位。
> `shape`:数据形状
> `order`:{'C', 'F'}, 数组排列方式“C”表示Ｃ语言风格，“Ｆ”表示Fortran风格。

```python
import numpy as np

data = np.arange(12, dtype='float32').resize((3, 4))

from tempfile import mkdtemp
import os.path as path

filename = path.join(mkdtemp(), 'newfile.dat')

fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
fp

#> memmap([[ 0.,  0.,  0.,  0.],
#>         [ 0.,  0.,  0.,  0.],
#>         [ 0.,  0.,  0.,  0.]], dtype=float64)
```

# 2. 参考资料

[numpy.memmap](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html)
