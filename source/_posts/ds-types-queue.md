---
type: algorithm
title: 数据结构与算法：队列类型
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-09-04 13:15:12
password:
summary:
tags: [数据结构, queue]
categories: 数据结构与算法
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

本文介绍不同类型的队列数据结构。

<!--more-->

# 1. 简介

队列就像排队买票，先到先得。有四种不同的队列：

- 简单队列（simple queue）
- 循环队列（circular queue）
- 优先队列（priority queue）
- 双端队列（double ended queue，deque）

# 2. 简单队列

在一个简单的队列中，插入发生在后面，移除发生在前面。 它严格遵循 FIFO（先进先出）规则。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904233021.png)

更详细内容，查看 [数据结构与算法：队列（queue）](https://rogerspy.github.io//2021/09/05/ds-queue/)。

# 3. 循环队列

循环队列是指，最后一个元素指向第一个元素，形成一个循环链。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904233239.png)

与简单队列相比，循环队列的主要优点是更好的内存利用率。 如果最后一个位置已满而第一个位置为空，我们可以在第一个位置插入一个元素。 此操作在简单队列中是不可能的。

更详细的内容，查看 [数据结构与算法：循环队列（circular-queue）](https://rogerspy.github.io/2021/09/05/ds-circular-queue/)。

# 4. 优先队列

优先级队列是一种特殊类型的队列，其中每个元素都与一个优先级相关联，并根据其优先级进行处理。 如果出现具有相同优先级的元素，则按照它们在队列中的顺序进行处理。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904233542.png)

更详细的内容，查看 [数据结构与算法：优先队列（priority queue）](https://rogerspy.github.io/2021/09/05/ds-priority-deque/)。

# 5. 双端队列

在双端队列中，可以从前面或后面执行元素的插入和删除。 因此，它不遵循 FIFO（先进先出）规则。

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904235805.png)

更详细的内容，查看 [数据结构与算法：双端队列（deque）](https://rogerspy.github.io/2021/09/05/ds-deque/)。

# Reference

[Types of Queues](https://www.programiz.com/dsa/types-of-queue) 
