---
type: algorithm
title: 数据结构与算法：算法简介
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-04-12 18:01:48
password:
summary:
tags: [数据结构]
categories: 数据结构与算法 
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

在开始学习算法之前先说一些废话。

## 1. 一个算法拯救无数生命

第二次世界大战期间，德军使用 [AM](https://en.wikipedia.org/wiki/AM_broadcasting) 进行信息交流，任何掌握对应 AM 频率和摩斯码的人都可以对信号进行解码得到信息。但是由于信息是被加密的，所以需要对信息进行解密。有时候人们很幸运能够猜对，但是很快德军又换了密码。

<!--more-->

阿兰·图灵加入英军，帮助他们对德军密码进行破译。最初他们建造了一台机器进行计算，但是由于计算过于耗时，所以最开始这台计算机并没有多大用处。后来，图灵改变了原来的算法使得解码速度大幅提升。在后来的战争中，这台机器帮助英军破译了大量的德军密码，大大加速了战争的结束。

*同一台机器从一个没什么用的废品，摇身一变成为拯救万民于水火的救世主，这就是算法的力量。*

另一个很有说服力的例子是 “*PageRank*” 算法。`PageRank` 帮助谷歌从一众搜索引擎中脱颖而出，成为行业领军者，因为 `PageRank` 能够得到更好的搜索结果。

## 2. 什么是算法？

算法就是计算机在完成任务过程中执行的步骤。就像人做菜，按照菜谱的步骤一步一步把各种调料和菜品烹饪成一道可口的菜，这个菜谱就是人在炒菜的时候的算法。对于计算机来说，算法就是一系列用于完成特定任务的指令。算法接收一系列输入，然后得到我们想要的输出。比如：

> 两个数字相加算法：
>
> - 输入两个数字
> - 使用 `+` 运算符对两个数字进行相加
> - 输出结果

算法也有“好的”算法和“不好的”算法，就像菜有好吃和不好吃一样。好的算法速度更快，不好的算法更慢。如果算法运算太慢意味着更高的成本，甚至在一些任务上是无法完成运算的（相对人的生命）。

# 3. 为什么学习算法？

计算机程序最宝贵的两种资源是时间和空间（内存）。

## 3.1 时间和空间（内存）都是有限的

程序运行时间用下式计算：
$$
t = t_e \times n
$$
其中 $n$ 表示指令数目，$t_e$ 表示运行每条指令需要的时间。$n$ 依赖于你编写的程序，而 $t_e$ 依赖于计算机硬件。

假设我们要计算 $1-10^{11}$ 的自然数之和。如果我们逐个数字相加，则需要 $10^{11}$ 次取值和 $10^{11}$ 次相加，即 $2\times 10^{11}$ 次计算。假设计算机每秒计算 $10^8$ 次，要计算完这个程序，仍需 16 分钟。

有没有什么办法帮助提高运算效率？我们中学就学过一个等差数列求和公式：
$$
\text{sum} = \frac{n\times (n+1)}{2}
$$
利用这个公式，我们只需要执行一次指令就可完成运算。

同样计算机内存并不是无限的，当你需要处理或者存储大量数据的时候，处理算法就需要考虑如何节省内存的使用。

## 3.2 可扩展性

可扩展性意味着算法或系统处理更大规模的问题的能力。

假设我们现在要建一间 50 人的教室，最简单的方法是订一个房间，放上一个黑板，几支粉笔，几张桌子和椅子就好了。如果学生是 200 人呢？我们还可以用上面的方法，只是需要用到更大的房间，更多的桌椅。但是如果学生是 1000 人呢？我们就找不到足够大的房间和足够多的的桌椅了。这个时候原来的方法就失效了。

就像前面的例子，最初我们计算实数和的方法就是不可扩展的，因为随着数据量增大，需要消耗的时间已经超出了我们所能忍受的范围。

而第二种方法就是可扩展的，因为无论数据量有多大，我们都能在一次指令下完成计算。

# 4. 结语

软件系统每天都会诞生许多新的方法和技术，而算法更像是其中的灵魂。当一个系统遇到瓶颈的时候，很可能通过优化其中的算法使整个系统性能得到提升。

但是同时也要注意，改善算法提升系统的可扩展性并不是唯一的方法。比如我们还可以通过分布式计算来实现。

#  Reference

[Why Learn Data Structures and Algorithms?](https://www.programiz.com/dsa/why-algorithms)
