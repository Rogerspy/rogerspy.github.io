---
type: algorithm
title: 数据结构与算法：队列（queue）
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-09-04 22:21:26
password:
summary:
tags: [数据结构]
categories: 数据结构与算法
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

本文介绍队列数据结构，并用 Python 代码实现。

<!--more-->

# 1. 简介

队列是一个非常有用的数据结构。它与电影院外排队买票是一样的，先排先买。队列也是如此，遵循先进先出（First In First Out，FIFO）原则。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904222908.png)

如上图所示，1 排在 2 前面，也会在 2 之前被删除。

在编程的术语中，将元素添加进队列中的操作叫做 “*enqueue*”，从队列中删除的操作叫做 “*dequeue*”。

# 2. 队列的基本操作

- **Enqueue**：向队列中添加元素
- **Dequeue**：从队列中删除元素
- **IsEmpty**：判断队列是否为空
- **IsFull**：判断队列是否为满队列
- **Peek**：获取队列最前面的元素而不删除该元素

<img src="https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/Queue-program-enqueue-dequeue.png" style="zoom: 50%;" />

1. 定义两个指针 `FRONT` 和 `REAR`
2. `FRONT` 追踪队列中第一个元素
3. `REAR` 追踪队列中最后一个元素 
4. 初始化 `FRONT` 和 `REAR` 都为 -1

## 2.1 Enqueue 操作

- 检查队列是否为满序列
- 对于第一个元素，设置 `FRONT` 为 0
- `REAR` 索引加 1
- 在 `REAR` 指向的位置处添加新元素

## 2.2 Dequeue

- 检查队列是否为空
- 返回 `FRONT` 指向的元素
- `FRONT` 的索引加 1
- 对于最后一个元素，重新设置 `FRONT` 和 `REAR` 为  -1

# 3. Python 实现队列

```python
# Queue implementation in Python

class Queue:

    def __init__(self):
        self.queue = []

    # Add an element
    def enqueue(self, item):
        self.queue.append(item)

    # Remove an element
    def dequeue(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    # Display  the queue
    def display(self):
        print(self.queue)

    def size(self):
        return len(self.queue)


q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
q.enqueue(5)

q.display()

q.dequeue()

print("After removing an element")
q.display()
```

# 4. 队列的限制

如下图所示，经过一系列的入队和出队，队列的尺寸减小了。但是我们只能在队列重置（所有的元素都出队）的时候设置 0 和 1 索引。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904230231.png)

对于队列的一种变种——循环队列来说，由于入队时尾指针向前追赶头指针；出队时头指针向前追赶尾指针，造成队空和队满时头尾指针均相等。因此，无法通过条件front==rear来判别队列是"空"是"满"。

# 5. 队列的时间复杂度

Enqueue 和 dequeue 操作在使用数组实现的队列中复杂度都是 $O(1)$。如果你用python中的 `pop(n)` 方法，那时间复杂度可能是 $O(n)$，取决于你要删除的元素的位置。

# 6. 队列的应用

- CPU  调度，硬盘调度。
- 当两个进程之间异步传输数据时，队列用于消息同步。
- 处理实时系统的中断。
- 呼叫中心电话系统使用队列将呼叫他们的人按顺序排列。

# Reference

[Queue Data Structure](https://www.programiz.com/dsa/queue) 
