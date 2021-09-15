---
type: algorithm
title: 数据结构与算法：数据结构简介
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-04-16 22:29:51
password:
summary:
tags: [数据结构, queue]
categories: 数据结构与算法
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

# 1. 什么是数据结构

数据结构是用来存储和组织数据的仓库，是一种计算机高效获取和更新数据的方式。根据你的项目需求，找到合适的数据结构至关重要。比如，你想存储序列数据，那么你可以将数据存储在 `Array` 数据结构中：

<!--more-->

<img width='500' src='https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/array_dsa.png'>

# 2. 数据结构类型

基本上，数据结构有两种类型：

- 线性数据结构
- 非线性数据结构

## 2.1 线性数据结构

线性数据结构中，数据是以一个数字接一个数字排列的形式组织起来的。典型的线性数据结构有：

### 2.1.1 数组 Array

数组数据结构中，数据排列在连续内存中，所有的元素都有相同的数据类型。而数据的具体形式由不同的编程语言决定，比如 `Python` 中，可以用 `list` 和 `array` 来实现。

<img width='400' src='https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/array_.png'>

### 2.1.2 堆栈 Stack

在堆栈数据结构中，数据以 *LIFO* 的原则进行存储，即后存进去的数据会先被删除。就像堆盘子，最后放上去的盘子会首先被拿下来。

<img width='250' src='https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/stack_dsa.png'>

### 2.1.3 队列 Queue

与堆栈数据结构相反，队列数据结构是以 *FIFO* 原则进行数据存储，即先进先出。就像排队，先排队的人先取到票。

<img width='350' src='https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/queue_dsa.png'>

### 2.1.4 链表 Linked List

链表中的每个元素都可以通过一系列的节点与其他元素相连，并且每个节点都包含元素本身以及下一个节点的地址。

<img width='700' src='https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/linked-list_dsa.png'>

## 2.2 非线性数据结构

非线性数据结构中的数据是无序的，且一个元素可以与其他元素相连。非线性数据结构主要有两种：

- 图数据结构
- 树数据结构

### 2.2.1 图数据结构

图数据结构中，每个节点称为“顶点”（*vertex*），顶点与顶点通过“边”相连。

<img src="https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/graph_dsa.png" alt="graph_dsa.png" style="zoom:50%;" />

常见的图数据结构包括：

- Spanning Tree and Minimum Spanning Tree
- Strongly Connected Components
- Adjacency Matrix
- Adjacency List

### 2.2.2 树数据结构

与图类似，树也是由顶点和边将数据组合在一起的。但是在树结构中，两个顶点只能有一条边。

<img width='350' src='https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/tree_dsa.png'>

常见的树数据结构有：

- Binary Tree
- Binary Search Tree
- AVL Tree
- B-Tree
- B+ Tree
- Red-Black Tree

## 2.3 线性数据结构 vs 非线性数据结构

| 线性数据结构                 | 非线性数据结构               |
| ---------------------------- | ---------------------------- |
| 数据按照一定顺序组织起来     | 数据是无序的                 |
| 数据只有一层                 | 数据有多个层级               |
| 只有一条路径可以遍历所有数据 | 通常需要多条路径遍历所有数据 |
| 内存使用率不高               | 不同的结构有不同的内存使用率 |
| 时间复杂度随数据量线性增加   | 时间复杂度保持不变           |



# Reference

[Data Structure and Types](https://www.programiz.com/dsa/data-structure-types) 
