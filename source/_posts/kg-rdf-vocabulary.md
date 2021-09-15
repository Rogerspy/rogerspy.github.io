---
type: blog
title: 知识图谱：知识建模（三）RDFS/OWL 词汇表
top: false
cover: false
toc: true
mathjax: true
body:
  - article
  - comments
gitalk:
  id: /wiki/material-x/
date: 2021-08-26 22:11:35
password:
summary:
tags: [KG, knowledge-modelling]
categories: 知识图谱
---

![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/kgicon.png)

前面的文章介绍了知识建模，我们提到知识建模使用的是 RDF 知识表示法，而 RDFS 本质上是一个标准化语义词汇表。所以本文总结一些常用的 RDFS/OWL 的语义词汇。

<!--more-->

# 0. 絮絮叨叨

## 0.1 RDF/XML

在正式介绍 RDFS/OWL 词汇之前，相信很多小伙伴在看知识建模的时候就会有很多疑问。为什么一定要用 RDF？XML 好像也能胜任这份工作，RDF 和 XML 的区别是什么？RDF 标榜的让计算机理解语义体现在哪里？等等一系列的疑问。当然回答这些问题并不是本文的目的，本文只是总结 RDFS/OWL 的词汇。要想弄明白 RDF 到底是怎么一回事，这里推荐一些必读的书籍/文献，希望能帮助到有疑问的人。

- [ Where are the Semantics in the Semantic Web?](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=08AF5BC897E861F62FE3800830B02E22?doi=10.1.1.91.8164&rep=rep1&type=pdf), *Michael Uschold*
- [Why RDF model is different from the XML model](https://www.w3.org/DesignIssues/RDF-XML.html), *Tim Berners-Lee*
- [A developer's guide to semantic web](https://www.researchgate.net/publication/279393307_A_Developer%27s_Guide_to_the_Semantic_Web), *Liyang Yu*
- [XML+RDF——实现Web数据基于语义的描述](http://www-900.ibm.com/developerWorks/cn/xml/x-xmlrdf/index.shtml#authorname), *周竞涛、王明微*

## 0.2 IRI, URI, URL, URN 的区别

- URL

  **Uniform Resource Locator**，统一资源定位符。是用于在网络中传播和访问互联网资源的一个地址，只有通过特定的地址才能够访问的指定的资源或者网页，简而言之就是我们所说的网址，当然这样有些不太恰当，但是确实最容易被理解的，就如你必须通过 `https://www.baidu.com` 才能访问百度搜索页面，通过其他的链接都是不行的，这个地址就被称之为 URL。

- URI

  **Uniform Resource Identifier**，统一资源标识符。也可以理解为标识、定位资源的字符串。字符集仅限于 US-ASCII 中（额外加一个百分号 %）， 不包括某些保留字符。URI 可用作定位器、名称或两者。 如果 URI 是定位器，则它描述了资源的主要访问机制。 如果一个 URI 是一个名称，它通过给它一个唯一的名称来标识一个资源。 许多人容易将 URL 和 URI 两者混淆，其实两者非常相似，但是也有所不同。URL 包含相对地址和绝对地址，URI 就属于绝对地址，所以 URL 包含 URI，简单的举例就是很多的网站可能都有 `/about` 这个路径，但是不同的域名或者 IP 访问到的就是不同的资源页面，所以这就只是一个标识，并不能标识其具体未知或者唯一性。

- IRI

  **Internationalized Resource Identifier**，国际化资源标识符。和 URI 类似，区别在于 URI 使用的字符集有限制，所以没有办法兼容不同的文字语言，所以 IRI 就引入了 Unicode 字符来解决这个兼容问题，最后就有了国际化资源标识符（IRI）。

- URN

  **Uniform Resource Name**，统一资源名称。旨在用作持久的，与位置无关的资源标识符。URN 可以提供一种机制，用于查找和检索定义特定命名空间的架构文件。尽管普通的 URL 可以提供类似的功能，但是在这方面，URN 更加强大并且更容易管理，因为 URN 可以引用多个 URL。子凡举个最简单的例子大家就明白了，那就是：磁力链接，它就是 URN 的一种实现，可以持久化的标识一个 BT 资源，资源分布式的存储在 P2P 网络中，无需中心服务器用户即可找到并下载它。

总结一下：

<div class='container' style='margin-top:40px;margin-bottom:20px;'>
    <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
        <div style='margin-left:10px'>
            <font color='white' size=4>
                • SUMMARY
            </font>
        </div>
    </div>
    <div style='background-color:#F3F4F7'>
        <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
            • IRI ⊃ URI <br />
            • URI ⊃ URL <br />
            • URI ⊃ URN <br />
            • URL ∩ URN = ∅ 
        </div>    
    </div>    
</div>

下面就进入今天的正题——RDFS/OWL 词汇表。本文摘自 *《语义网技术体系》 [瞿裕忠，胡伟，程龚 编著] 2015年版* 这本书。想查看完整版词汇表，可前往这两个网页：

- [RDF Schema 1.1](https://www.w3.org/TR/rdf-schema/) 
-  [OWL Web Ontology Language](https://www.w3.org/TR/2004/REC-owl-ref-20040210/)

再啰嗦一句，除了以上两个 RDF 词汇表，还有一个 [FOAF](http://xmlns.com/foaf/spec/) 词汇表在对人物进行建模的时候通常会用到。但是这里就不再介绍，想了解更多可自行前往。

# 1. 序言

RDF Schema（下文简称 RDFS） 是 RDF 词汇表的一个扩展版本（RDF 本身是一个知识表示模型，但同时也是一个词汇表）。RDFS 承认有许多技术可以用来描述类和属性的含义，例如 OWL。

本文中定义的语言由一组 RDF 资源组成，这些资源可用于在特定于应用程序的 RDF 词汇表中描述其他 RDF 资源。核心词汇 `rdfs` 非正式地称为命名空间中定义。该命名空间由 IRI 标识：

```
http://www.w3.org/2000/01/rdf-schema#
```

并且通常与前缀相关联 `rdfs:`。本规范还使用前缀 `rdf:`来指代 RDF 命名空间：

```
http://www.w3.org/1999/02/22-rdf-syntax-ns#
```

为了方便和可读性，本规范使用缩写形式来表示 IRI。形式 `prefix:suffix` 的名称应该被解释为一个 IRI，它由与后缀连接的前缀相关联的 IRI 组成。

# 2. RDFS

资源可以被分成不同的组，这些组称之为“类”（classes）。每个类别下包含的成员称之为“实例”。比如“人”是一个类，“张三”是一个“人”的实例。通常我们把 RDF 和 RDFS 合写成 RDF(S) 或 RDF/S。

下面分别介绍 RDF(S) 的核心词汇。

## 2.1 Classes

资源可以分成不同的组，这些组就称之为“类”，组内的成员就称之为类的“实例”。我们用 IRI 来标识类，然后用 RDF 属性来描述类。两个不同的类可能有相同的实例，比如“张三”既可以是“导演”这个类，也可以是“演员”这个类。一个类也可能是他自己的实例。

> 名词解释：“类的外延”
>
> 与一个类别相关的集合，我们称之为类的外延。类的外延集合中的每个成员都是类的实例。举个例子：
>
> 类：食物
>
> 类的外延：a = {鸡，鸭，鱼，肉}
>
> 类的实例：鸡，鸭，鱼，肉
>
> 例子中，“食物”作为一个类别，表示一个抽象概念。跟这个类别相关的一个集合 a 表示“食物”的外延，相对类来说类的外延是具体的概念。但是要注意 a 作为一个集合整体出现。而 a 中的每一个元素称之为实例。
>
> 当我们说“鸡肉是一种食物”的时候，实际上是表明“鸡肉”是“食物”这个概念的外延集合中的一员。
> $$
> \text{instance} \in a \rightarrow class
> $$

### 2.1.1 rdf:Resource

所有 RDF 描述的事物都是资源，即都是 `rdfs:Resource` 的实例。这是所有事物的类，其他所有类都是它的子类。`rdfs:Resource` 也是 `rdfs:Class` 的实例。

### 2.1.2 rdf:Class

对应“类”的概念，即资源的类。当定义一个新类的时候，表示该类的资源必须有一个 `rdf:type` 属性，属性值是 `rdfs:Class`。比如定义“导演”是一个新类，那么我们必须定义：

```
导演 rdf:type rdfs:Class
```

注意，如上所述，一个实例可能属于多个类，所以类成员是不互斥的。`rdfs:Class` 是 `rdfs:Class` 是实例。

### 2.1.3 rdf:Literal

`rdf:Literal` 表示类或属性的字面量类型，比如数字、字符串等。`rdfs:Literal` 是 `rdfs:Class` 的实例，同时也是 `rdfs:Resource` 的子类。

### 2.1.4 rdfs:Property

`rdfs:Property` 是 RDF 属性类，同时也是 `rdfs:Class` 的实例。

## 2.2.  Properties

在 RDF 中，RDF 属性表示 subject 资源和 object 资源之间的关系。为了下文解释方便，我们这里写下三元组的一般形式：

```
subject predicate object
```

## 2.2.1 rdfs:range

`rdfs:range` 是 `rdfs:Property` 的一个实例，用来指明一个属性的值域。例如三元组：

```
p rdfs:range c
```

表示 p 是 `rdfs:range` 的一个实例， c 是 `rdfs:Class`  的一个实例。上面的三元组描述的是一个 predicate 是 p 的 object 是 c 的实例。

### 2.2.2 rdfs:domain

`rdfs:domain` 是 `rdfs:Property` 的一个实例，用来指明一个属性的定义域。例如三元组：

```
p rdfs:domain c
```

表示 p 是 `rdfs:Property` 的一个实例，c 是 `rdfs:Class` 的实例。上面的三元组描述的是一个 predicate 是 p 的 subject 是 c 的实例。

其中，如果 p 有不止一个 `rdfs:domain` ，那么其对应的所有 subject 都是 c 的实例。

举个例子：

```
人 吃 食物  

吃 rdf:type rdfs:Property
吃 rdfs:domain 人
吃 rdfs:range 食物
```

翻译过来就是，“吃”表示一种属性（关系），它的主语是“人”，宾语是“食物”。

### 2.2.3 rdf:type

`rdf:type` 是 `rdf:Property` 的一个实例，用于描述一个资源是类的实例，例如：

```
R rdf:type C 
```

表示 C 是 `rdfs:Class` 的子类，并且 R 是 C 的实例。用一句通俗易懂的话就是，R 是一种 C，比如 `人 rdf:type 生物` 表示“人是一种动物”。实际上 `rdf:type` 表示 “is-a” 的关系，可以简写成 `a`。

### 2.2.4 rdfs:subClassOf

`rdfs:subClassOf` 是 `rdfs:Property`  的一个实例，用来指明一个类的所有实例也是另一个类的实例，比如：

```
C1 rdfs:subClassOf C2
```

描述的是，C1 是 `rdfs:Class` 的一个实例，C2 是 `rdfs:Class` 的一个实例，并且 C1 是 C2 的一个子类。`rdfs:subClassOf` 是可传递的，即如果 a 是 b 的子类，b 是 c 的子类，那么 a 也是 c 的子类。

`rdfs:subClassOf` 的 `rdfs:domain` 是 `rdfs:Class`。`rdfs:subClassOf` 的 `rdfs:range` 是 `rdfs:Class`。

### 2.2.5 rdfs:subPropertyOf

`rdfs:subPropertyOf` 是 `rdfs:Property` 的一个实例，用来指明与一个资源相关的所有属性也与另一个资源相关，比如：

```
P1 rdfs:subPropertyOf P2
```

描述了 P1 是 `rdfs:Property` 的一个实例，P2 也是 `rdfs:Property` 的一个实例，并且 P1 是 P2 的一个子属性。`rdfs:subPropertyOf` 是可传递性的。

`rdfs:subPropertyOf` 的 `rdfs:domain` 是 `rdf:Property`。`rdfs:subPropertyOf` 的 `rdfs:range` 是 `rdf:Property`。

除了上面介绍的词之外， RD(S) 还有很多其他有用的词汇，这里不一一列举。下图展示了 RDF(S) 各个词汇之间的关系：

 ![](https://cdn.jsdelivr.net/gh/rogerspy/blog-imgs/blog-imgs/20210830172323.png)

### 2.2.6 RDFS 词汇总结

#### 2.2.6.1 Classes

| Class name                                                   | comment                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [rdfs:Resource](https://www.w3.org/TR/rdf-schema/#ch_resource) | The class resource, everything.                              |
| [rdfs:Literal](https://www.w3.org/TR/rdf-schema/#ch_literal) | The class of literal values, e.g. textual strings and integers. |
| [rdf:langString](https://www.w3.org/TR/rdf-schema/#ch_langstring) | The class of language-tagged string literal values.          |
| [rdf:HTML](https://www.w3.org/TR/rdf-schema/#ch_html)        | The class of HTML literal values.                            |
| [rdf:XMLLiteral](https://www.w3.org/TR/rdf-schema/#ch_xmlliteral) | The class of XML literal values.                             |
| [rdfs:Class](https://www.w3.org/TR/rdf-schema/#ch_class)     | The class of classes.                                        |
| [rdf:Property](https://www.w3.org/TR/rdf-schema/#ch_property) | The class of RDF properties.                                 |
| [rdfs:Datatype](https://www.w3.org/TR/rdf-schema/#ch_datatype) | The class of RDF datatypes.                                  |
| [rdf:Statement](https://www.w3.org/TR/rdf-schema/#ch_statement) | The class of RDF statements.                                 |
| [rdf:Bag](https://www.w3.org/TR/rdf-schema/#ch_bag)          | The class of unordered containers.                           |
| [rdf:Seq](https://www.w3.org/TR/rdf-schema/#ch_seq)          | The class of ordered containers.                             |
| [rdf:Alt](https://www.w3.org/TR/rdf-schema/#ch_alt)          | The class of containers of alternatives.                     |
| [rdfs:Container](https://www.w3.org/TR/rdf-schema/#ch_container) | The class of RDF containers.                                 |
| [rdfs:ContainerMembershipProperty](https://www.w3.org/TR/rdf-schema/#ch_containermembershipproperty) | The class of container membership properties, rdf:_1, rdf:_2, ..., all of which are sub-properties of 'member'. |
| [rdf:List](https://www.w3.org/TR/rdf-schema/#ch_list)        | The class of RDF Lists.                                      |

#### 2.2.6.2 Properties

| Property name                                                | comment                                                | domain        | range         |
| ------------------------------------------------------------ | ------------------------------------------------------ | ------------- | ------------- |
| [rdf:type](https://www.w3.org/TR/rdf-schema/#ch_type)        | The subject is an instance of a class.                 | rdfs:Resource | rdfs:Class    |
| [rdfs:subClassOf](https://www.w3.org/TR/rdf-schema/#ch_subclassof) | The subject is a subclass of a class.                  | rdfs:Class    | rdfs:Class    |
| [rdfs:subPropertyOf](https://www.w3.org/TR/rdf-schema/#ch_subpropertyof) | The subject is a subproperty of a property.            | rdf:Property  | rdf:Property  |
| [rdfs:domain](https://www.w3.org/TR/rdf-schema/#ch_domain)   | A domain of the subject property.                      | rdf:Property  | rdfs:Class    |
| [rdfs:range](https://www.w3.org/TR/rdf-schema/#ch_range)     | A range of the subject property.                       | rdf:Property  | rdfs:Class    |
| [rdfs:label](https://www.w3.org/TR/rdf-schema/#ch_label)     | A human-readable name for the subject.                 | rdfs:Resource | rdfs:Literal  |
| [rdfs:comment](https://www.w3.org/TR/rdf-schema/#ch_comment) | A description of the subject resource.                 | rdfs:Resource | rdfs:Literal  |
| [rdfs:member](https://www.w3.org/TR/rdf-schema/#ch_member)   | A member of the subject resource.                      | rdfs:Resource | rdfs:Resource |
| [rdf:first](https://www.w3.org/TR/rdf-schema/#ch_first)      | The first item in the subject RDF list.                | rdf:List      | rdfs:Resource |
| [rdf:rest](https://www.w3.org/TR/rdf-schema/#ch_rest)        | The rest of the subject RDF list after the first item. | rdf:List      | rdf:List      |
| [rdfs:seeAlso](https://www.w3.org/TR/rdf-schema/#ch_seealso) | Further information about the subject resource.        | rdfs:Resource | rdfs:Resource |
| [rdfs:isDefinedBy](https://www.w3.org/TR/rdf-schema/#ch_isdefinedby) | The definition of the subject resource.                | rdfs:Resource | rdfs:Resource |
| [rdf:value](https://www.w3.org/TR/rdf-schema/#ch_value)      | Idiomatic property used for structured values.         | rdfs:Resource | rdfs:Resource |
| [rdf:subject](https://www.w3.org/TR/rdf-schema/#ch_subject)  | The subject of the subject RDF statement.              | rdf:Statement | rdfs:Resource |
| [rdf:predicate](https://www.w3.org/TR/rdf-schema/#ch_predicate) | The predicate of the subject RDF statement.            | rdf:Statement | rdfs:Resource |
| [rdf:object](https://www.w3.org/TR/rdf-schema/#ch_object)    | The object of the subject RDF statement.               | rdf:Statement | rdfs:Resource |

# 3. OWL

由于 RDFS 的表达能力较弱，W3C 2004 年又发布了 Web Ontology Language（OWL）进一步提供更加丰富的知识表示和推理能力。OWL 以描述逻辑为理论基础，可以将概念和属于用结构化的形式表示出来。通过 RDF 中的链接可以是本体分布在不同的系统中，充分体现了其标准化，开放性，扩展性以及适应性。现在 OWL 已经是 W3C 推荐的本体建模标准。OWL 的命名空间是：

```
http://www.w3.org/2002/07/owl#
```

OWL 提供 3 中表达能力不同的子语言：OWL Full，OWL DL，OWL Lite。其中任意一个都可以映射成一个完整的 RDF 图。

- OWL Full。完全兼容 RDFS，但超出经典一阶逻辑的范畴。与 OWL Full 相关的推理工具现在还在探索中。
- OWL DL。是 OWL Full 的一个子集，表达能力相对较强，可以有效的支持逻辑推理，但不是完全兼容 RDFS。
- OWL Lite。在 OWL DL 的基础上对允许使用公理做了进一步的限制。

到了 2012 年，W3C 对原先版本的 OWL 进行了修订，发布新的 OWL 版本——OWL 2。OWL 2 对 OWL 向后兼容，包含了 3 个指定的概图：

- OWL 2 EL。允许以高效的多项式时间算法对类型的可满足性检查、分类和实例检查并进行推理，特别适合使用含有大量属性或类本体的应用。
- OWL 2 QL。允许使用传统的关系数据库实现查询问答，特别适合使用大量实例数据并且以查询问答作为主要推理任务的应用。
- OWL 2 RL。允许以一种比较直接的方式，使用基于规则的推理引擎，在不牺牲太多的表达能力的情况下实现大规模推理。

## 3.1 OWL Document

一般情况下，描述本体的文档都包含本体本身的信息。一个本体是一个资源，可以采用 OWL 和其他命名空间属性进行描述。这些描述被称为本体头部，通常位于本体文档的开始部分。

```
<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.semanticweb.org/qiuji/ontologies/2017/9/untitled-ontology-2#"
     xml:base="http://www.semanticweb.org/qiuji/ontologies/2017/9/untitled-ontology-2"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:untitled-ontology-22="http://www.semanticweb.org/ontologies/2017/9/untitled-ontology-2#"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:untitled-ontology-2="http://www.semanticweb.org/qiuji/ontologies/2017/9/untitled-ontology-2#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://www.semanticweb.org/ontologies/2017/9/untitled-ontology-2"/>
```

### 3.1.1 owl:imports

允许引用另一个包含定义的 OWL 本体，并将其含义作为定义本体的一部分。每个引用都包含一个 URI，它指向被导入的本体。

```
@prefix : <http://example.com/pwl/families/> .
@prefix otherOnt: <http/example.org/otherOntologies/families/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>

<http://example.com/pwl/families/>
    rdf:type owlOntology ;
    owl:imports <http/example.org/otherOntologies/families.owl> .
```

另外，还可以在本体头部添加有关版本的一些信息。相关属性包括：`owl:versionInfo`，`owl:priorVersion`，`owl:backwardCompatibleWith`，`owl:incompatibleWith` 等等。

## 3.2 OWL Classes

与 RDFS 类似， OWL 也有“类”的概念，也是表示我们对资源的分组，也有“类的外延”等概念。需要注意的是，在OWL 中，类的外延中的元素称之为个体（individual），和在 Protege 建模工具菜单栏中的 individual 是同一概念，都表示实例。

<div class='container' style='margin-top:40px;margin-bottom:20px;'>
    <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
        <div style='margin-left:10px'>
            <font color='white' size=4>
                • NOTE
            </font>
        </div>
    </div>
    <div style='background-color:#F3F4F7'>
        <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
            OWL DL 和 OWL DL 中的资源不能同时是个体（individual）和类（class），即 class 和 individual 是互斥的。
            另外，rdfs:Class 和 rdfs:Property 是被禁止使用的。
        </div>    
    </div>    
</div>

从上面的介绍来看，OWL 被设计出来主要是对 RDFS  的逻辑推理能力进行补强。要进行推理我们首先要有一些公理。在 OWL 中采用“类描述”对 OWL 类进行解释描述，然后将 OWL 组合成类公理。

### 3.2.1 类描述（class description）

类描述通过类名或通过指定未命名匿名类的类外延来描述 OWL 类。OWL 中有 6 中不同的类描述：

1. 类标识符（URI）
2. 穷举组成一个类的个体（enumeration）
3. 属性限制（property restriction）
4. 多个类描述的交集（intersection）
5. 多个类描述的并集（union）
6. 一个类描述的补集（complement）

类标识符相当于通过类名（URI）来描述一个类；穷举表示一个类包含可穷举的个体；一个类中的所有个体都要满足特定的属性限制。对于 4、5、6 来说，可以认为是逻辑与（AND）或（OR）非（NOT）操作。

#### 3.2.1.1 owl:Class

`owl:Class` 表示一个明明资源是一个类别，比如：

```
ex：Human rdf:type owl:Class .
```

其中 `ex` 表示本体的命名空间。下面的例子我们都用 RDF/XML 语法进行举例，所以上面的例子改写成：

```
<owl:Class rdf:ID="Human"/>
```

<div class='container' style='margin-top:40px;margin-bottom:20px;'>
    <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
        <div style='margin-left:10px'>
            <font color='white' size=4>
                • NOTE
            </font>
        </div>
    </div>
    <div style='background-color:#F3F4F7'>
        <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
            OWL Lite 和 OWL DL 中 <code>owl:Class</code> 必须用在所有的类描述上。
        </div>    
    </div>    
</div>

<div class='container' style='margin-top:40px;margin-bottom:20px;'>
    <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
        <div style='margin-left:10px'>
            <font color='white' size=4>
                • NOTE
            </font>
        </div>
    </div>
    <div style='background-color:#F3F4F7'>
        <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
            在 OWL Lite 和 OWL DL 中<code>owl:Class</code> 是 <code>rdfs:Class</code> 的子类。这个关系说明
            在 RDFS 中，并不是所有的类在 OWL DL(Lite) 都是合法的。但是在 OWL Full 中二者是等价的。
        </div>    
    </div>    
</div>

OWL 类标识符是预先定义好的，即 `owl:Thing` / `owl:Nothing`。`owl:Thing` 是所有 OWL 类的父类，而 `owl:Nothing` 是所有类的子类（可以认为就是空集）。

#### 3.2.1.2 owl:oneOf

`owl:oneOf` 用来表示类描述中的穷举，它的值必须是类的实例。为了方便，我们可以用 `rdfs:parseType="Collection"` ，例如：

```
<owl:Class>
  <owl:oneOf rdf:parseType="Collection">
    <owl:Thing rdf:about="#Eurasia"/>
    <owl:Thing rdf:about="#Africa"/>
    <owl:Thing rdf:about="#NorthAmerica"/>
    <owl:Thing rdf:about="#SouthAmerica"/>
    <owl:Thing rdf:about="#Australia"/>
    <owl:Thing rdf:about="#Antarctica"/>
  </owl:oneOf>
</owl:Class>
```

<div class='container' style='margin-top:40px;margin-bottom:20px;'>
    <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
        <div style='margin-left:10px'>
            <font color='white' size=4>
                • NOTE
            </font>
        </div>
    </div>
    <div style='background-color:#F3F4F7'>
        <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
            OWL Lite 没有穷举。
        </div>    
    </div>    
</div>

#### 3.2.1.3 owl:Restriction

属性限制是一类特殊的类描述。它用来描述所有个体都满足一定限制条件的匿名类。OWL 有两种属性限制：值限制和基数限制。

- 所谓值限制指的是，限制属性的值域。
- 所谓基数限制指的是，限制属性的个数。

OWL 还提供了全局基数限制：`owl:FunctionalProperty` 和 `owl:InverseFunctionalProperty`。

`owl:Restriction` 是 `owl:Class` 的子类。一个限制类应该有一个三元组用 `owl:onProperty` 来连接属性和限制。

- **值限制**

  1. `owl:allValuesFrom`：用来限制一个类的所有个体是否在指定的值域内。比如：

     ```
     <owl:Restriction>
       <owl:onProperty rdf:resource="#hasParent" />
       <owl:allValuesFrom rdf:resource="#Human"  />
     </owl:Restriction>
     ```

  2. `owl:someValuesFrom`：用来限制一个类的所有个体中，至少有一个个体来源于指定的值域。比如：

     ```
     <owl:Restriction>
       <owl:onProperty rdf:resource="#hasParent" />
       <owl:someValuesFrom rdf:resource="#Physician" />
     </owl:Restriction>
     ```

  3. `owl:hasValue`：用来限制一个类的所有个体中，至少有一个（语义上）等于指定的值。比如：

     ```
     <owl:Restriction>
       <owl:onProperty rdf:resource="#hasParent" />
       <owl:hasValue rdf:resource="#Clinton" />
     </owl:Restriction>
     ```

     <div class='container' style='margin-top:40px;margin-bottom:20px;'>
         <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
             <div style='margin-left:10px'>
                 <font color='white' size=4>
                     • NOTE
                 </font>
             </div>
         </div>
         <div style='background-color:#F3F4F7'>
             <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
                 “语义上等价于”的意思是，V 不一定是指定的值，但是 V 和指定的值 V1 之间有一个 <code>owl:sameAs</code>的关系。
             </div>    
         </div>    
     </div>

- **基数限制**

  1. `owl:maxCardinality`：用来限制一个类包含了最多 N 个语义不同的个体，其中 N 就是基数限制的值。比如：

     ```
     <owl:Restriction>
       <owl:onProperty rdf:resource="#hasParent" />
       <owl:maxCardinality rdf:datatype="&xsd;nonNegativeInteger">2</owl:maxCardinality>
     </owl:Restriction>
     ```

  2. `owl:minCardinality`：用来限制一个类至少包含 N 个语义不同的个体，其中 N 就是基数限制的值。

     比如：

     ```
     <owl:Restriction>
       <owl:onProperty rdf:resource="#hasParent" />
       <owl:minCardinality rdf:datatype="&xsd;nonNegativeInteger">2</owl:minCardinality>
     </owl:Restriction>
     ```

     <div class='container' style='margin-top:40px;margin-bottom:20px;'>
         <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
             <div style='margin-left:10px'>
                 <font color='white' size=4>
                     • NOTE
                 </font>
             </div>
         </div>
         <div style='background-color:#F3F4F7'>
             <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
                 一个类中的所有实例都要有 N 个属性。
             </div>    
         </div>    
     </div>

  3. ` owl:cardinality`：用来限制一个类必须要有 N 个语义不同的个体，不能多也不能少。其中 N 就是基数限制的值。比如：

     ```
     <owl:Restriction>
       <owl:onProperty rdf:resource="#hasParent" />
       <owl:cardinality rdf:datatype="&xsd;nonNegativeInteger">2</owl:cardinality>
     </owl:Restriction>
     ```

#### 3.2.1.4 Intersection, union and complement

- `owl:intersectionOf`：连接一个类和一个类描述的列表，表示这个类的外延中的个体同时也是列表中所有类描述的外延成员。比如：

  ```
  <owl:Class>
    <owl:intersectionOf rdf:parseType="Collection">
      <owl:Class>
        <owl:oneOf rdf:parseType="Collection">
          <owl:Thing rdf:about="#Tosca" />
          <owl:Thing rdf:about="#Salome" />
        </owl:oneOf>
      </owl:Class>
      <owl:Class>
        <owl:oneOf rdf:parseType="Collection">
          <owl:Thing rdf:about="#Turandot" />
          <owl:Thing rdf:about="#Tosca" />
        </owl:oneOf>
      </owl:Class>
    </owl:intersectionOf>
  </owl:Class>
  ```

  `owl:intersectionOf` 可以看成逻辑连词。

- `owl:unionOf`：表示一个个体至少会出现在列表中的一个类中。比如：

  ```
  <owl:Class>
    <owl:unionOf rdf:parseType="Collection">
      <owl:Class>
        <owl:oneOf rdf:parseType="Collection">
          <owl:Thing rdf:about="#Tosca" />
          <owl:Thing rdf:about="#Salome" />
        </owl:oneOf>
      </owl:Class>
      <owl:Class>
        <owl:oneOf rdf:parseType="Collection">
          <owl:Thing rdf:about="#Turandot" />
          <owl:Thing rdf:about="#Tosca" />
        </owl:oneOf>
      </owl:Class>
    </owl:unionOf>
  </owl:Class>
  ```

- `owl:complementOf`：连接一个类和一个类描述，表示类外延中的个体不属于类描述的外延。比如：

  ```
  <owl:Class>
    <owl:complementOf>
      <owl:Class rdf:about="#Meat"/>
    </owl:complementOf>
  </owl:Class>
  ```

### 3.2.2 类公理

类描述通过类公理组合在一起用来定义一个类。这句话说起来很拗口，其实描述的道理很简单。就相当于我们要炒一盘菜，需要一些原材料（类描述），然后通过一些原则（类公理）将这些原料组合在一起形成一盘菜（类）。

OWL 提供了 3 个词汇，将类描述组合起来：

- `rdfs:subClassOf`
- `owl:equivalentClass`
- `owl:disjointWith`

#### 3.2.2.1 rdfs:subClassOf

$$
class\ description \quad \text{rdfs:subClassOf} \quad class\ description
$$

这里的 `rdfs:subClassOf` 和 RDFS 中的一样。比如：

```
<owl:Class rdf:ID="Opera">
  <rdfs:subClassOf rdf:resource="#MusicalWork" />
</owl:Class>
```

#### 3.2.2.2 owl:equivalentClass

$$
class\ description\quad \text{owl:equivalentClass}\quad class\ description
$$

` owl:equivalentClass` 表示两个类描述有相同的类外延。最简单的形式是，两个命名类别是等价的。比如：

```
<owl:Class rdf:about="#US_President">
  <equivalentClass rdf:resource="#PrincipalResidentOfWhiteHouse"/>
</owl:Class>
```

<div class='container' style='margin-top:40px;margin-bottom:20px;'>
    <div style='background-color:#54c7ec;height:36px;line-height:36px;vertical-align:middle;'>
        <div style='margin-left:10px'>
            <font color='white' size=4>
                • NOTE
            </font>
        </div>
    </div>
    <div style='background-color:#F3F4F7'>
        <div style='padding:15px 10px 15px 20px;line-height:1.5;'>
            <code>owl:equivalentClass</code> 的两个类并不表示两个类是等价的。
        </div>    
    </div>    
</div>

比如上例中，“美国总统” 这个概念和“白宫的主要居民”这个概念并不一样。真正的语义等价应该用 `owl:sameAs`。

#### 3.2.2.3 owl:disjointWith

$$
class\ description\quad \text{owl:disjointWith}\quad class\ description
$$

`owl:disjointWith` 表示两个类描述没有公共的个体，或者说两个类描述是互斥的。比如：

```
<owl:Class rdf:about="#Man">
  <owl:disjointWith rdf:resource="#Woman"/>
</owl:Class>
```

## 3.3 Properties

 OWL 有两种属性：对象属性（object property）和数据类型属性（datatype property）。对象属性用来连接两个实例，而数据类型属性连接一个实例和寿哥数据类型的字面量。换成比较容易理解的话就是，对象属性表示两个实体之间的关系，数据类型属性就是实体和属性之间的关系。比如

```
小明 父亲 大明
小明 生日 1990/1/1
```

其中“父亲”就是对象属性，“生日”就是数据类型属性。

OWL 中支持的属性机构包括：

- RDFS ：`rdfs:subPropertyOf`, `rdfs:domain` 和 `rdfs:range`
- 与其他属性相关的： `owl:equivalentProperty` 和 `owl:inverseOf`
- 全局基数限制：`owl:FunctionalProperty` 和 `owl:InverseFunctionalProperty`
- 逻辑属性： `owl:SymmetricProperty` 和 `owl:TransitiveProperty`

### 3.3.1 owl:equivalentProperty

`owl:equivalentProperty` 表示两个属性有相同的属性外延。类似 `owl:equivalentClass`。

### 3.3.2 owl:inverseOf

属性是有方向的，从定义域指向值域。`owl:inverseOf` 表示反向属性，即原属性的定义域和值域互换。比如：

```
<owl:ObjectProperty rdf:ID="hasChild">
  <owl:inverseOf rdf:resource="#hasParent"/>
</owl:ObjectProperty>
```

### 3.3.3 owl:FunctionalProperty

`owl:FunctionalProperty` 表示对于实例 $x$ 来说，只有唯一的 $y$ 值。比如：

```
<owl:ObjectProperty rdf:ID="husband">
  <rdfs:domain rdf:resource="#Woman" />
  <rdfs:range  rdf:resource="#Man" />
</owl:ObjectProperty>

<owl:FunctionalProperty rdf:about="#husband" />
```

### 3.3.4 owl:InverseFunctionalProperty

`owl:InverseFunctionalProperty` 表示与 `owl:FunctionalProperty` 相反的意思，即对于值 $y$ 只能有一个 实例 $x$ 与之对应。比如：

```
<owl:InverseFunctionalProperty rdf:ID="biologicalMotherOf">
  <rdfs:domain rdf:resource="#Woman"/>
  <rdfs:range rdf:resource="#Human"/>
</owl:InverseFunctionalProperty>
```

### 3.3.5 owl:TransitiveProperty

`owl:TransitiveProperty` 表示属性的可传递性。如果 $(x,y)$ 是 P 的实例，$(y,z)$ 也是 P 的实例，那么 $(x,z)$ 也是 P 的实例。比如：

```
<owl:TransitiveProperty rdf:ID="subRegionOf">
  <rdfs:domain rdf:resource="#Region"/>
  <rdfs:range  rdf:resource="#Region"/>
</owl:TransitiveProperty>
```

### 3.3.6 owl:SymmetricProperty

`owl:SymmetricProperty` 表示如果 $(x,y)$ 是 P 的实例，那么 $(y,x)$ 也是 P 的实例。比如：

```
<owl:SymmetricProperty rdf:ID="friendOf">
  <rdfs:domain rdf:resource="#Human"/>
  <rdfs:range  rdf:resource="#Human"/>
</owl:SymmetricProperty>
```

## 3.4 Individuals

个体分为两种：

1. 类的成员和个体的属性值
2. 个体身份

### 3.4.1 类的成员和个体属性值

```
<Opera rdf:ID="Tosca">
  <hasComposer rdf:resource="#Giacomo_Puccini"/>
  <hasLibrettist rdf:resource="#Victorien_Sardou"/>
  <hasLibrettist rdf:resource="#Giuseppe_Giacosa"/>
  <hasLibrettist rdf:resource="#Luigi_Illica"/>
  <premiereDate rdf:datatype="&xsd;date">1900-01-14</premiereDate>
  <premierePlace rdf:resource="#Roma"/>
  <numberOfActs rdf:datatype="&xsd;positiveInteger">3</numberOfActs> 
</Opera>
```

### 3.4.2 个体身份

通常我们会给不同的事物取不同的名字，但是我们并不能保证不重名。比如“苹果”既可以是电子产品，也可以是水果。为了对个体的身份进行区分或合并，OWL 也设计了一套词汇：

- `owl:sameAs`：表明是相同的个体，只是名字不同
- `owl:differentFrom`：表明是两个不同的个体
- `owl:AllDifferent`：表明列表中所有的个体都不相同

```
<rdf:Description rdf:about="#William_Jefferson_Clinton">
  <owl:sameAs rdf:resource="#BillClinton"/>
</rdf:Description>
```

```
<Opera rdf:ID="Don_Giovanni"/>

<Opera rdf:ID="Nozze_di_Figaro">
  <owl:differentFrom rdf:resource="#Don_Giovanni"/>
</Opera>

<Opera rdf:ID="Cosi_fan_tutte">
  <owl:differentFrom rdf:resource="#Don_Giovanni"/>
  <owl:differentFrom rdf:resource="#Nozze_di_Figaro"/>
</Opera>
```

```
<owl:AllDifferent>
  <owl:distinctMembers rdf:parseType="Collection">
    <Opera rdf:about="#Don_Giovanni"/>
    <Opera rdf:about="#Nozze_di_Figaro"/>
    <Opera rdf:about="#Cosi_fan_tutte"/>
    <Opera rdf:about="#Tosca"/>
    <Opera rdf:about="#Turandot"/>
    <Opera rdf:about="#Salome"/>
  </owl:distinctMembers>
</owl:AllDifferent>
```

# 4. 结语

关于 RDFS 和 OWL 的词汇总结我们就介绍这么多。当然，这些都只是一小部分，要想看完整版的推荐看 w3c 的官方文档。我们总结出来的这些词汇是比较常用的，同时也是有助于帮助不了解本体，不了解知识建模的同学对这些东西有一个大体的概念。其实本体建模就是在构建一套逻辑体系，这套逻辑体系帮助计算机进行逻辑推理。而无论是 RDFS 还是 OWL 亦或是其他众多我们没有介绍的词汇表都是在尝试将这样一个逻辑体系进行标准化。先阶段计算机的逻辑推理能力仍然处于很弱的阶段，说明我们现在的工作仍然很初级。我们这里总结的相关内容也许在不久的将来就会过期，失效甚至被推翻。但是了解这些知识也有助于我们对未来的发展有一个清晰的认知。

# Reference

1.  [RDF Schema 1.1](https://www.w3.org/TR/rdf-schema/)
2.  [OWL Web Ontology Language](https://www.w3.org/TR/2004/REC-owl-ref-20040210/) 
3.  [IRI, URI, URL, URN and their differences](https://fusion.cs.uni-jena.de/fusion/blog/2016/11/18/iri-uri-url-urn-and-their-differences/), *JAN MARTIN KEIL* 
4.  [浅谈什么是 URL、URI、IRI、URN 及之间的区别](https://zhangzifan.com/t/7393.html), *张子凡* 
5.  语义网技术体系 [瞿裕忠，胡伟，程龚 编著] 2015年版
6.  [知识图谱-浅谈RDF、OWL、SPARQL](https://www.jianshu.com/p/9e2bfa9a5a06), *吕不韦* 

