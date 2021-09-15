---
type: algorithm
title: 算法与数据结构（Python）：array
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-08-09 22:21:24
password:
summary:
tags: [数组]
categories: 数据结构与算法
---

数组是一个容器，它容纳的元素应该是相同的数据类型。数组有两个重要概念：

- **元素** —— 存储的数组中的数据称为元素。
- **索引** —— 数组中每个元素所在的位置。

<!--more-->

# 1. 数组的表示

![](https://codingdict.com/static/assets/tutorials/python/ds/array_declaration.jpg)

- `int`  表示数组中数字的类型为整型
- `array` 表示数组的名字
- `[10]` 表示数组的尺寸，即数组中有多少个元素
- `{35, 33, 42, ...}` 表示数组存储的数据

![](https://codingdict.com/static/assets/tutorials/python/ds/array_representation.jpg)

- 索引从 0 开始
- 数组的尺寸是 10，表示它可以存储 10 个元素
- 每个元素可以通过索引访问

# 2. 基本操作

数组的基本操作包括：

- **遍历** —— 逐个获得数组中的元素
- **插入** —— 在指定的位置（索引）处添加一个元素
- **删除** —— 删除指定位置（索引）处的元素
- **搜索** —— 搜索指定位置（索引）处的元素
- **更新** —— 更新指定位置（索引）处的元素

`python` 内置的 `array` 模块可以用来创建数组：

```python
from array import *

arrayName = array(typecode, [Initializerss])
```

其中 `typecode` 用于定义数组中元素的数据类型，一些常用的 `typecode` 如下：

| typecode | 表示                         |
| :------: | :--------------------------- |
|    b     | 大小为1字节/ td>的有符号整数 |
|    B     | 大小为1字节的无符号整数      |
|    C     | 大小为1字节的字符            |
|    i     | 大小为2个字节的带符号整数    |
|    I     | 大小为2个字节的无符号整数    |
|    F     | 大小为4字节的浮点            |
|    d     | 大小为8个字节的浮点          |

举个例子：

```python
from array import *

array1 = array('i', [10,20,30,40,50])
```

##  2.1 遍历

```python
for x in array1:
    print(x)
```

输出

```python
10
20
30
40
50
```

## 2.2 搜索

```python
# 方法一
print(array1[0])
print(array1[2])

# 方法二
print(array1.index(40))
```

输出

```python
10
30
3
```

## 2.3 插入

```python
array1.insert(1,60)
print(array1)
```

输出

```python
array('i', [10, 60, 20, 30, 40, 50])
```

## 2.4 删除

```python
array1.remove(40)
print(array1)
```

输出

```python
array('i', [10, 60, 20, 30, 50])
```

## 2.5 更新

```python
array1[2] = 80
print(array1)
```

输出

```python
array('i', [10, 60, 80, 30, 50])
```

# Reference

[Python-数组](https://codingdict.com/article/4830)
