---
type: article
title: Data Structures With Python
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-07-27 23:41:30
password:
summary:
tags: Data Structure
categories: 博客转载
---

![](https://mymasterdesigner.com/wp-content/uploads/2021/07/Data-Structures-With-Python-Big-Guide.png)

对于编程和计算机科学来说，数据结构是主要的主题，几乎涉及所有计算机领域。

本文介绍 `python` 中的一些数据结构。

<!--more-->

# 1. 什么是数据结构

计算机科学中，数据结构是一种为了便于数据获取和修改的组织、管理和存储的形式。所有编程语言中，列表、字典和数组是最简单的数据结构。尽管语法不同，但是其内在的逻辑是相同的。因此，本文介绍的例子也适用于其它编程语言。

# 2. 字典、映射、哈希表

`Python` 中的字典（`dictionary`）可以用来存储任意数据，每条数据都有一个关键词。映射（`map`）也被称为哈希表（`hash table`）、查找表（`lookup table`）或者关联数组（`associative array`）。它可以更轻松地组织与特定关键字关联的数据，并以更有条理的形式呈现。

比如，用字典来存储每个人的年龄：

```python
data = {
    'Mark': 12,
    'Alice': 23,
    'David': 8
}
```

当我们想要查看特定的人的年龄时：

```python
data['Mark']

# Output: 12
```

> 当然，你可以把数据写在同一行内，但是如果数据量比较大的时候，卸载一行看起来会比较乱。

## 2.1 `OrderedDict`, `defaultdict`, `ChainMap`

- 字典是无序的，如果我们想按照顺序来存储数据，显然原生的字典就无能为力了，这个时候就可以用 `OrderedDict`：

```python
import collections as cs

dict1 = cs.OrderedDict(
    Mark=12,
    Alice=23,
    David=8
)
```

查看一下 `dcit1`：

```python
print(dict1)

# Ouput: ([('Mark', 12), ('Alice', 22), ('David', 8)])
```

- 当我们从字典里面取值的时候，遇到字典里并没有对应的 key 的时候，程序就会报错。这时 `defaultdict` 就派上用场了。`defaultdict` 的作用是在于，当字典里的key不存在但被查找时，返回的不是 `keyError` 而是一个默认值。

```python
from collections import defaultdict

dict1 = defaultdict(int)
dict2 = defaultdict(set)
dict3 = defaultdict(str)
dict4 = defaultdict(list)
```

`defaultdict` 接受 `int`, `set`, `str`, `list` 作为参数，也可以自定义函数作为参数。我们来看下，上面的例子中默认值分别是什么：

```python
print(dict1[1]) 
print(dict2[1])
print(dict3[1])
print(dict4[1])

# Output:
0
set()

[]
```

说明 `int` 默认值是 0，`set` 默认值是空集合，`str` 默认值是空字符串，`list` 默认值是空列表。

- 当你有多个字典的时候，可以使用 `ChainMap` 将它们变成一个字典。

```python
from collections import ChainMap

dict1 = {"1": 1, "2": 2}
dict2 = {"3": 3, "4": 4}
main = ChainMap(dict1, dict2)

print(main["3"] , main["1"])

# Output:
3 1
```

# 3. 数组型数据结构

## 3.1 列表

列表可以存储任意类型的数据。

```python
# from 0 to 10 value
arr = [1,2,3,4,5,6,7,8,9,10]

# String Array
arr1 = ["a" , "b" , "c"]

# Get First Indexing
arr1[0]

# Get from 0 to 4
arr[0:4]

# Deleting Element
del arr[0]

# Adding Element
arr.append(11)
```

```python
print(a)

# Output:  [1, 2, 3, 4]
```

## 3.2 元组

元组是另一个可以存储任意类型数据的数据结构。与列表不同的是，元组是不可变的。

```python
tuple = (1 , 2 , 3)
tuple[0]

# Output: 1

tuple1 = ("x" , 1 , 1.25)
tuple1[2]

# Output: 1.25

# you'll get error
del tuple[0]
tuple[1] = "y"

# 报错
```

## 3.3 `array` 数组

`python` 的 `array` 模块存储的数据包括整型、浮点型等等，它的空间占用率会更低。因为 `array` 只接受相同类型的数据。

```python
# Accessing array
from array import array

# use type code
arr = array("f" , [1.0 , 1.2])
```

## 3.4 字符串——字符编码的数组

字符串可以节省空间，因为它们被密集打包并专门处理特定的数据类型。 如果要保存 `Unicode` 文本，则应使用字符串。

```python
str = "55555"
emoji = "😀"

print(str , emoji)

# Outtput: 55555 😀
```

## 3.5 字节 & 字节数组

字节（*Bytes*）存储的是 0 到 255 的数字，如果超过了这个范围程序会报错。

```python
x = bytes([1 , 2 , 3])
y = bytes([-1 , 2 , 3])
z = bytes([100 , 200 , 300])
```

```python
Output: b'\x01\x02\x03'
Output: error
Output: error
```

# 4. 集合 & 多数组数据结构

## 4.1 集合

集合中不能包含相同的数据，且集合存储的是无序的数据。

```python
set = {1 , 2 , 3}

set.add(4)
set.remove(3)
```

## 4.2 Frozen Set

原始的集合元素可增可删，如果我们不想让集合发生改变，可以使用 `frozenset` 方法：

```python
frozen = frozenset({"x" , "y" , "z"})
frozen.add("k")

# 报错
```

## 4.3 Counter

`counter` 可以对多个集合进行合并，并且可以对每个元素进行计数，得到每个元素的个数。

```python
from collections import Counter

merge = Counter()

fruits = {"apple" , "banana" , "orange"}
merge.update(fruits)
print(merge)

fruits1 = {"apple" , "banana" , "watermelon"}
merge.update(fruits1)
print(merge)
```

```python
{'orange': 1, 'apple': 1, 'banana': 1})
{'apple': 2, 'banana': 2, 'orange': 1, 'watermelon': 1})
```

# 5. 堆栈

堆栈是支持用于插入和删除的快速输入/输出语义 (LIFO) 的项目集合。与数组和列表不同，你不能做随机访问，你需要使用函数进行插入和删除。

## 5.1 `list` 实现堆栈

你可以用 `append` 把数据加到最后，再用`pop`从 LIFO 队列中取出。

```python
stack = []

stack.append(1)
stack.append(2)
stack.append(3)

print(stack)

stack.pop()
stack.pop()

print(stack)
```

```python
# Output: [1,2,3]
# Output: [1]
```

## 5.2 `deque` 实现堆栈

`deque` 与列表的区别还支持在固定时间添加和删除数组开头的元素。

因此，它比列表更有效、更快。 它还支持随机访问。

如果您尝试删除双端队列之间的数据，您可能会失去性能，主要原因是直到两端的所有元素都移动以腾出空间或填补空白。

```python
from collections import deque

stack = deque()

stack.append("a")
stack.append("b")
stack.append("c")

print(stack)

print(stack.pop())
print(stack.pop())

print(stack)
```

```python
# Output: deque(['a', 'b', 'c'])
# Output: deque(['a'])
```

# 6. `Queues`

堆栈上的逻辑在这里略有不同，其中采用先进先出 (FIFO)，而在堆栈中采用先进后出。

我们这里可以使用栈中使用的 `list`和 `deque` 数据结构，也可以使用队列中的 `Queue` 类。

```python
queue = []

queue.append("x")
queue.append("y")
queue.append("z")

print(queue)

queue.pop(0)
queue.pop(0)

print(queue)
```

```python
# Output: ['x', 'y', 'z']
# Output: ['z']
```

## 6.1 `deque`

```python
from collections import deque

queue = deque()

queue.append("x")
queue.append("y")
queue.append("z")

print(queue)

queue.popleft()
queue.popleft()

print(queue)
```

```python
# Output: deque(['x', 'y', 'z'])
# Output: deque(['z'])
```

##  6.2 `queue`

队列是一种结构，通过它我们可以确定队列可以容纳和存储多少数据。 它主要用于实现队列。

您可以通过将 `max size` 参数设置为 0 来创建无限队列，这符合 FIFO 规则。

```python
from queue import Queue

queue = Queue(maxsize = 0)

# Adding element
queue.put(10)
queue.put(20)
queue.put(30)

print(queue.queue)

# Removing element
queue.get()
queue.get()

print(queue.queue)
```

```python
# Output: deque([10, 20, 30])
# Output: deque([30])
```

# 7. 自定义数据类型

要更可控，您只需要您自己。 不要害怕创建和使用自己的类。 创建复杂的类有时会很累，但它会提高您的工作效率。

当您想通过方法向记录对象添加业务逻辑和活动时，创建自定义类是一个极好的解决方案。 然而，这意味着这些东西不再只是数据对象。

```python
class Student:
    def __init__(self, name, note):
        self.name = name
        self.note = note

x = Student("David" , 55)
y = Student("Mark" , 35)

# Access Data
print(x.name , x.note)

print(Student)
```

```python
# Output: David 55
# Output: <main.Student object at 0x7f53925c2400>
```

# 8. Reference

[Data Structures With Python – Big Guide](https://mymasterdesigner.com/2021/07/06/data-structures-with-python-big-guide/)

