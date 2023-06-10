title: Entity Linking
date: 2020-01-02
mathjax: true
categories:
- AI
- NLP
tags:
- NLP
- Entity Linking
---

Entity Linking

<!-- more --> 

- Knowledge Graph (知识图谱)：一种语义网络，旨在描述客观世界的概念实体及其之间的关系，有时也称为Knowledge Base (知识库)。
    - 图谱由三元组构成：`<实体1，关系，实体2>` 或者 `<实体，属性，属性值>`；
    - 例如：`<姚明，plays-in，NBA>`、`<姚明，身高，2.29m>`；
    - 常见的KB有：Wikidata、DBpedia、YAGO。
- Entity 实体：实体是知识图谱的基本单元，也是文本中承载信息的重要语言单位。
- Mention 提及：自然文本中表达实体的语言片段。

应用方向

1. **Question Answering**：EL是KBQA的刚需，linking到实体之后才能查询图数据库；
2. **Content Analysis**：舆情分析、内容推荐、阅读增强；
3. **Information Retrieval**：基于语义实体的搜索引擎，google搜索一些实体，右侧会出现wikipedia页面；
4. **Knowledge Base population**：扩充知识库，更新实体和关系。

候选实体和消歧

Entity linking system consists of two components: 

1. candidate entity generation：从mention出发，找到KB中所有可能的实体，组成候选实体集 (candidate entities)；
2. Entity Disambiguation：从candidate entities中，选择最可能的实体作为预测实体。


# Entity Disambiguation (ED)

是最重要的部分

- Features
    - Context-Independent Features：
        - LinkCount：#(m->e)，知识库中某个提及m指向实体e的次数；
        - Entity Attributes：Popularity、Type；
    - Context-Dependent Features：
        - Textual Context：BOW, Concept Vector
        - Coherence Between Entities：WLM、PMI、Jaccard Distance

## Context-Independent Features

mention到实体的LinkCount、实体自身的一些属性（比如热度、类型等等）

- LinkCount作为一个先验知识，在消歧时，往往很有用

## Context-Dependent Features

全局地进行entities的消歧实际上是一个NP-hard的问题，因此核心问题是如何更加快速有效地利用一致性特征

- **Learning to Rank Methods**：Point-wise、Pair-wise、List-wise。由于ED任务ground truth只有一个实体，一般都是用**point-wise**来做。输入是文本的context、mention、某个entity的一些attributes，输出mention指向该entity的置信度，以此rank，选出最可信的entity；
- **Probabilistic Methods**：Incorporate heterogeneous knowledge into a probabilistic model。结合不同信息，得到条件概率  $P(e|m,c)$，其中 c 是输入文本，e 为实体， m 是mention。比如用归一化的LinkCount信息，作为先验概率 $P(e|m)$ ；
- **Graph-Based Approaches**：maximize coherene between entities。利用图特征 (entity embedding、relation)，在消歧时，考虑全局消歧后实体的一致性；

### Deep Type

[Discovering Types for Entity Disambiguation](https://openai.com/blog/discovering-types-for-entity-disambiguation/)

## **High-level overview**

Our system uses the following steps:

1. **Extract every Wikipedia-internal link to determine, for each word, the set of conceivable entities it can refer to.** For example, when encountering the link `[jaguar](https://en.wikipedia.org/wiki/Jaguar)` in a Wikipedia page, we conclude that `https://en.wikipedia.org/wiki/Jaguar` is one of the meanings of `jaguar`.
2. **Walk the Wikipedia category tree (using the [Wikidata](https://www.wikidata.org/wiki/Wikidata:Introduction) knowledge graph) to determine, for each entity, the set of categories it belongs to.** For example, at the bottom of `https://en.wikipedia.org/wiki/Jaguar_Cars`’s Wikipedia page, are the following categories (which themselves have their own categories, such as [Automobiles](https://en.wikipedia.org/wiki/Category:Automobiles)):

    ![https://openai.com/content/images/2018/02/jaguar.png](https://openai.com/content/images/2018/02/jaguar.png)

3. **Pick a list of ~100 categories to be your “type” system, and optimize over this choice of categories so that they compactly express any entity.** We know the mapping of entities to categories, so given a type system, we can represent each entity as a ~100-dimensional binary vector indicating membership in each category.
4. **Using every Wikipedia-internal link and its surrounding context, produce training data mapping a word plus context to the ~100-dimensional binary representation of the corresponding entity, and train a neural network to predict this mapping.** This chains together the previous steps: Wikipedia links map a word to an entity, we know the categories for each entity from step 2, and step 3 picked the categories in our type system.
5. **At test time, given a word and surrounding context, our neural network’s output can be interpreted as the probability that the word belongs to each category.** If we knew the exact set of category memberships, we would narrow down to one entity (assuming well-chosen categories). But instead, we must play a probabilistic 20 questions: use [Bayes’ theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) to calculate the chance of the word disambiguating to each of its possible entities.

## Unlinkable Mention Prediction 拒识掉未知实体

- **NIL Threshold**：通过一个置信度的阈值来卡一下；
- **Binary Classification**：训练一个二分类的模型，判断Top-rankeded Entity是否真的是文中的mention想要表达的实体；
- **Rank with NIL**：在rank的时候，在候选实体中加入NIL Entity。

一般就阈值卡一下就好了，不是太大的问题。但如果具体的场景是做KB Population且实体还不是很全的时候，就需要重点关注一下了。

## **Candidate Entity Generation (CEG)**

CEG的方法都比较朴素

- 最重要的方法：Name Dictionary ( `{mention: entity}` )
- 哪些别名：首字母缩写、模糊匹配、昵称、拼写错误等。
- 构建方法：
    - Wikipedia（Redirect pages, Disambiguation pages, Hyperlinks）；
    - 基于搜索引擎：调google api，搜mention。若前m个有wiki entity，建立map；
    - Heuristic Methods；
    - 人工标注、用户日志。

对于每一个entity，紧凑而充分地配置别名，才能保证生成的candidate entites没有遗漏掉ground truth entity。

具体的，要配置哪些别名，要用什么构建方法，往往取决于EL的使用场景。比如做百科问答或是通用文本的阅读增强，就很依赖于**wikipedia和搜索引擎**；但如果是某个具体的行业领域，就需要通过一些**启发式的方法、用户日志、网页爬取，甚至人工标注的方法**来构建Name Dictionary。

# Reference
- [【知识图谱】实体链接：一份"由浅入深"的综述](https://zhuanlan.zhihu.com/p/100248426)
- [Discovering Types for Entity Disambiguation](https://openai.com/blog/discovering-types-for-entity-disambiguation/)
