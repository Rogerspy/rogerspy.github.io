---
type: algorithm
title: 数据结构与算法：双端队列（deque）
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-09-05 01:12:57
password: 12345678
summary:
tags: [数据结构, queue]
categories: 数据结构与算法
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

本文介绍双端队列，并用 Python 实现。

<!--more-->

# 1. 简介

双端队列（deque），顾名思义指的是队列的前端和后端都可以进行插入和删除。因此，它不遵循 FIFO 原则。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905144730.png)

双端队列有两种：

- 输入限制型双端队列：这种队列中输入被限制在一端，而删除则可以两端同时进行；
- 输出限制型双端队列：这种队列只能在一端进行删除，而插入元素则可以两端同时进行。

# 2. 双端队列的基本操作

下面我们以循环队列实现的双端队列为例尽心介绍。在循环队列中，如果队列是满的，那么我们就从头开始。

但是，用线性队列实现的双端队列中，如果队列是满的，队列就不允许再往里插入元素了。

展示双端队列基本操作之前，我们有一些预备工作：

1. 设置队列的大小 `n`；
2. 定义两个指针 `FRONT=-1` 和 `REAR=0`。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905150638.png)

## 2.1 在头部插入元素

1. 检查前端位置

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905181644.png)

2. 如果 `FRONT < 1`，重置 `FRONT=n-1`（最后一个索引）

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905181844.png)

   

3. 否则， `FRONT-1`

4. 在 `FRONT` 的位置添加新元素

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905193549.png)

## 2.2 在尾部添加元素

1. 检查队列是否是满队列

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905193906.png)

2. 如果队列是满的，重置 `REAR=0`

3. 否则，`REAR=REAR+1`

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905194245.png)

4. 在 `REAR` 的位置添加新元素

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905194423.png)

## 2.3 从头部删除元素

1. 检查队列是否为空

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905194650.png)

2. 如果队列是空（即 `FRONT=-1`），不能删除元素。

3. 如果队列只有一个元素（即 `FRONT=REAR`），设置 `FRONT=-1` 以及 `REAR=-1`。

4. 否则如果 `FORNT=n-1`，则令 `FRONT`去到首位，即令 `FRONT=0`。

5. 否则 `FRONT=FORNT+1`。

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905195221.png)

## 2.4 从尾部删除元素

1. 检查队列是否为空。

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905195459.png)

2. 如果队列为空（`FRONT=-1`），不能删除元素。

3. 如果队列只有一个元素（即 `FRONT=REAR`），设置 `FRONT=-1` 以及 `REAR=-1`。

4. 如果 `REAR` 在前面（`REAR=0`），则令 `REAR=n-1`。

5. 否则 `REAR=REAR-1`。

   ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905195900.png)

## 2.5 检查队列是否为空

检查 `FRONT=-1`，如果为真则队列为空。

## 2.6 检查队列是否为满

如果 `FRONT=0` 以及 `REAR=n-1` 或者 `FRONT=REAR+1` 则队列为满。

# 3. Python 实现双端队列

```python
# Deque implementaion in python

class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def addRear(self, item):
        self.items.append(item)

    def addFront(self, item):
        self.items.insert(0, item)

    def removeFront(self):
        return self.items.pop(0)

    def removeRear(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


d = Deque()
print(d.isEmpty())
d.addRear(8)
d.addRear(5)
d.addFront(7)
d.addFront(10)
print(d.size())
print(d.isEmpty())
d.addRear(11)
print(d.removeRear())
print(d.removeFront())
d.addFront(55)
d.addRear(45)
print(d.items)
```

# 4. 时间复杂度

上述操作的时间复杂度为 $O(1)$。

# 5. 双端队列的应用

1. 软件的撤销操作
2. 浏览器存储浏览历史
3. 用来实现队列和栈

# Reference

[Deque Data Structure](https://www.programiz.com/dsa/deque) 
