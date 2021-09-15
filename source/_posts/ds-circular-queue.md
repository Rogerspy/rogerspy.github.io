---
type: algorithm
title: 数据结构与算法：循环队列（circular-queue）
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-09-05 00:00:33
password:
summary:
tags: [数据结构, queue]
categories: 数据结构与算法
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

本文介绍循环队列，并用 Python 实现循环队列。

<!--more-->

# 1. 简介

循环队列是常规队列（简单队列）的变种，是将队列中最后一个元素与第一个元素相连。因此，循环队列看起来像下图的样子：

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210905000636.png)

循环队列是为了解决简单队列的限制。在常规队列中，经过一系列的出队入队操作之后，会有一些空位置。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904230231.png)

上图中 0 和 1 的位置会被空置，除非等到队列重置。

# 2. 循环队列的基本操作 

循环队列通过循环递增的方式工作，即当我们递增指针并到达队列的末尾时，我们又从队列的开头开始。其中递增是通过模除队列的尺寸，即：

```
if REAR + 1 == 5 (overflow!), REAR = (REAR + 1)%5 = 0 (start of queue)
```

具体过程如下：

- 两个指针 `FRONT` 和 `REAR`
- `FRONT` 追踪队列中第一个元素
- `REAR` 追踪队列中最后一个元素
- 初始化 `FRONT` 和 `REAR` 为 -1

<img src="https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/circular-queue-program.png" style="zoom:50%;" />

## 2.1 Enqueue

- 检查队列是否是满队列
- 对于第一个元素，设置 `FRONT` 的值为 0
- 循环增加 `REAR` ，如果 `REAR` 到末尾，下一步就从头开始
- 在 `REAR` 指向的位置添加新元素

## 2.2 Dequeue

- 检查队列是否为空
- 返回 `FRONT` 指向的值
- `FRONT` 循环加 1
- 对于最后一个元素，重置 `FRONT` 和 `REAR` 为 -1

然而，检查满队列的时候，有一个新问题：

- 第一种情况：`FRONT=0` && `REAR=size-1`
- 第二种情况：`FRONT=REAR+1`

第二种情况下，`REAR` 因为循环递增而从 0  开始，并且其值只比 `FRONT` 时，队列已满。

# 3. Python实现循环队列

```python
# Circular Queue implementation in Python


class MyCircularQueue():

    def __init__(self, k):
        self.k = k
        self.queue = [None] * k
        self.head = self.tail = -1

    # Insert an element into the circular queue
    def enqueue(self, data):

        if ((self.tail + 1) % self.k == self.head):
            print("The circular queue is full\n")

        elif (self.head == -1):
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            self.tail = (self.tail + 1) % self.k
            self.queue[self.tail] = data

    # Delete an element from the circular queue
    def dequeue(self):
        if (self.head == -1):
            print("The circular queue is empty\n")

        elif (self.head == self.tail):
            temp = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return temp
        else:
            temp = self.queue[self.head]
            self.head = (self.head + 1) % self.k
            return temp

    def printCQueue(self):
        if(self.head == -1):
            print("No element in the circular queue")

        elif (self.tail >= self.head):
            for i in range(self.head, self.tail + 1):
                print(self.queue[i], end=" ")
            print()
        else:
            for i in range(self.head, self.k):
                print(self.queue[i], end=" ")
            for i in range(0, self.tail + 1):
                print(self.queue[i], end=" ")
            print()


# Your MyCircularQueue object will be instantiated and called as such:
obj = MyCircularQueue(5)
obj.enqueue(1)
obj.enqueue(2)
obj.enqueue(3)
obj.enqueue(4)
obj.enqueue(5)
print("Initial queue")
obj.printCQueue()

obj.dequeue()
print("After removing an element from the queue")
obj.printCQueue()
```

# 4. 循环队列时间复杂度

基于数组实现的循环队列，其 enqueue 和 dequeue 时间复杂度都是 $O(1)$。

# 5. 循环队列的应用

- CPU 任务调度
- 内存管理
- 任务堵塞管理

# Refernece

[Circular Queue Data Structure](https://www.programiz.com/dsa/circular-queue) 

 
