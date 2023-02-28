<h1>Pandas基础教程（十）向量化字符串操作</h1>

[TOC]

我们知道numpy最大的优势是向量化数据操作，避免了使用循环，从而大大提高效率。但是对于字符串来说就没这么幸运了，numpy并没有提供这种向量化的字符串操作。

Pandas补足了这一缺憾，pandas提供了向量化操作字符串的操作。而且避免了使用循环的时候带来的麻烦，比如缺失值。

```python
data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]
#> ['Peter', 'Paul', 'Mary', 'Guido']

data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
[s.capitalize() for s in data]
#> AttributeError: 'NoneType' object has no attribute 'capitalize'
```

可以看到如果列表中存在缺失值，`capitalize()`操作就会报错，如果要避免报错还需要加入判断语句，使得代码臃肿难以理解。这完全不符合python的编程美学。

```python
import pandas as pd
names = pd.Series(data)
names
#> 0    peter
#> 1     Paul
#> 2     None
#> 3     MARY
#> 4    gUIDO
#> dtype: object

names.str.capitalize()
#> 0    Peter
#> 1     Paul
#> 2     None
#> 3     Mary
#> 4    Guido
#> dtype: object
```

是不是世界突然变成了彩色的了？

# 1. pandas字符串方法

pandas提供了以下字符串方法：

<table>
<thead><tr>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>len()</code></td>
<td><code>lower()</code></td>
<td><code>translate()</code></td>
<td><code>islower()</code></td>
</tr>
<tr>
<td><code>ljust()</code></td>
<td><code>upper()</code></td>
<td><code>startswith()</code></td>
<td><code>isupper()</code></td>
</tr>
<tr>
<td><code>rjust()</code></td>
<td><code>find()</code></td>
<td><code>endswith()</code></td>
<td><code>isnumeric()</code></td>
</tr>
<tr>
<td><code>center()</code></td>
<td><code>rfind()</code></td>
<td><code>isalnum()</code></td>
<td><code>isdecimal()</code></td>
</tr>
<tr>
<td><code>zfill()</code></td>
<td><code>index()</code></td>
<td><code>isalpha()</code></td>
<td><code>split()</code></td>
</tr>
<tr>
<td><code>strip()</code></td>
<td><code>rindex()</code></td>
<td><code>isdigit()</code></td>
<td><code>rsplit()</code></td>
</tr>
<tr>
<td><code>rstrip()</code></td>
<td><code>capitalize()</code></td>
<td><code>isspace()</code></td>
<td><code>partition()</code></td>
</tr>
<tr>
<td><code>lstrip()</code></td>
<td><code>swapcase()</code></td>
<td><code>istitle()</code></td>
<td><code>rpartition()</code></td>
</tr>
</tbody>
</table>

下面我们举几个例子说明

```python
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
# 返回字符串
monte.str.lower()
#> 0    graham chapman
#> 1       john cleese
#> 2     terry gilliam
#> 3         eric idle
#> 4       terry jones
#> 5     michael palin
#> dtype: object

# 返回数字
monte.str.len()
#> 0    14
#> 1    11
#> 2    13
#> 3     9
#> 4    11
#> 5    13
#> dtype: int64

# 返回布尔值
monte.str.startswith('T')
#> 0    False
#> 1    False
#> 2     True
#> 3    False
#> 4     True
#> 5    False
#> dtype: bool

# 返回列表
monte.str.split()
#> 0    [Graham, Chapman]
#> 1       [John, Cleese]
#> 2     [Terry, Gilliam]
#> 3         [Eric, Idle]
#> 4       [Terry, Jones]
#> 5     [Michael, Palin]
#> dtype: object
```

# 2. 正则表达式方法

|方法|描述|
|-|-|
|<code>match()</code>|每个元素上调用<code>re.match()</code>, 返回布尔值|
|<code>extract()</code>|每个元素上调用<code>re.match()</code>,　返回匹配到的字符串|
|<code>findall()</code>|每个元素上使用<code>re.findall()</code>|
|<code>replace()</code>|替换匹配到的模式|
|<code>contains()</code>|在每个元素上调用<code>re.search()</code>, 返回布尔值|
|<code>count()</code>|返回匹配到的个数|
|<code>split()</code>|等效于<code>str.split()</code>, 但是接收的是正则模式|
|<code>rsplit()</code>|等效于<code>str.rsplit()</code>，接收正则模式|

下面举几个例子：

```python
monte.str.extract('([A-Za-z]+)', expand=False)
#> 0     Graham
#> 1       John
#> 2      Terry
#> 3       Eric
#> 4      Terry
#> 5    Michael
#> dtype: object

monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
#> 0    [Graham Chapman]
#> 1                  []
#> 2     [Terry Gilliam]
#> 3                  []
#> 4       [Terry Jones]
#> 5     [Michael Palin]
#> dtype: object
```

# 3. 其他方法

|方法|描述|
|-|-|
|<code>get()</code>|按索引获取每一个元素|
|<code>slice()</code>|对每个元素进行切片|
|<code>slice_replace()</code>|用传入的值替代切片的内容|
|<code>cat()</code>|拼接字符串|
|<code>repeat()</code>|重复字符串|
|<code>normalize()</code>|返回字符串的Unicode形式|
|<code>pad()</code>|增加空格（前，后或两边）|
|<code>wrap()</code>|分割长字符串|
|<code>join()</code>|以指定的分隔符拼接字符串|
|<code>get_dummies()</code>|提取虚拟的（dummy）变量成dataframe|

其他的方法都比较容易理解，我们只介绍`get_dummies()`方法：

```python
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
full_monte                           
#>     info            name
#> 0  B|C|D  Graham Chapman
#> 1    B|D     John Cleese
#> 2    A|C   Terry Gilliam
#> 3    B|D       Eric Idle
#> 4    B|C     Terry Jones
#> 5  B|C|D   Michael Palin

full_monte['info'].str.get_dummies('|')
#>    A  B  C  D
#> 0  0  1  1  1
#> 1  0  1  0  1
#> 2  1  0  1  0
#> 3  0  1  0  1
#> 4  0  1  1  0
#> 5  0  1  1  1
```
