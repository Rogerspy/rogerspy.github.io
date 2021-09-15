---
type: algorithm
title: 数据结构与算法：栈（stack）
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-09-04 12:16:52
password:
summary:
tags: [数据结构, stack]
categories: 数据结构与算法
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/20210820161802.png)

本文介绍栈（stack）数据结构，并用 python 代码实现。

<!--more-->

# 1. 简介

栈是一个线性数据结构，遵循后进先出（Last In First Out，LIFO）的原则。这就意味着最后插入的元素会先被删除。

<img src="https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904122325.png" style="zoom:80%;" />

就像叠盘子一样，

- 你可以在最上面放一个新盘子
- 拿盘子的时候也是从最上面开始拿

如果想要最下面的盘子，你就必须先把上面所有的盘子先拿走。这就是栈的基本工作方式。

# 2. 栈的 LIFO 原则

用编程的术语来说，在栈最上面放置一个元素称之为 “*push*”，删除元素叫做 “*pop*”。

<img src="https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904122919.png" style="zoom: 80%;" />

# 3. 栈的基本操作

- **push**：在栈上面添加一个元素 
- **pop**：从栈中删除一个元素
- **isEmpty**：判断栈是否为空
- **isFull**：判断栈是否是一个满栈
- **peek**：获取栈最上层的元素而不删除它

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210904124034.png)

1. 用 `TOP` 指针来追踪栈中最上层的元素
2. 初始化栈的时候，我们设置设置指针为 -1，这样我们就可以通过判断 `TOP==-1` 来检查栈是否为空
3. 当往栈里 push 数据的时候，我峨嵋你增加 `TOP` 的值，将新元素放置在 `TOP` 指定的位置
4. 删除元素的时候，返回 `TOP` 指向的值，然后减小 `TOP` 值
5. 向栈 push 数据的时候应该先检查栈是否已满
6. 删除数据的时候，应该检查栈是否为空

# 4. 用 Python 实现栈

```python
# Stack implementation in python


# Creating a stack
def create_stack():
    stack = []
    return stack


# Creating an empty stack
def check_empty(stack):
    return len(stack) == 0


# Adding items into the stack
def push(stack, item):
    stack.append(item)
    print("pushed item: " + item)


# Removing an element from the stack
def pop(stack):
    if (check_empty(stack)):
        return "stack is empty"

    return stack.pop()


stack = create_stack()
push(stack, str(1))
push(stack, str(2))
push(stack, str(3))
push(stack, str(4))
print("popped item: " + pop(stack))
print("stack after popping an element: " + str(stack))
```

# 5. 栈的时间复杂度

对于基于数组的栈实现，push 和 pop 操作都是常数时间，即 $O(1)$。

# 6. 栈的应用

尽管栈是非常简单的数据结构，但是它非常有用，最常见的应用如：

- 词倒置。将词中的每个字符方法栈中，然后一个一个删除就可以了。因为栈是 LIFO 的，删除的时候就可以将词中的字符倒置过来。
- 编译器中，计算比如 `2+4/5*(7-9)` 的表达式的时候，用栈将表达式转化成前缀或者后缀的形式。
- 浏览器中，后退按钮用栈存储了所有你浏览过的网址（URL），每次你浏览一个新的网站的时候，它就会被加入到栈中，当你回退的时候，现在的网页就会被删除，然后回到倒数第二个页面。

# Reference

[Stack Data Structure](https://www.programiz.com/dsa/stack) 
