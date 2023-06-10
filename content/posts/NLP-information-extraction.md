title: 信息抽取
date: 2018-01-11
mathjax: true
categories:
- AI
- NLP
tags:
- NLP
- Information Retrieval
---
## 信息抽取
1997年MUC会议（MUC-7） 召开时，评测任务已经增加到5个：
① 场景模板（scenario template, ST）填充：定义了描述场景的模板及槽填充规范；
② 命名实体（named entity, NE）识别：识别出文本中出现的专有名称和有意义的数量短语， 并加以归类；
③ 共指（coreference, CR）关系确定：识别出给定文本中的参照表达（ referring expressions），并确定这些表达之间的共指关系；
④ 模板元素（template element, TE）填充：类似于人名和组织机构名识别，但是要求系统必须识别出实体的描述和名字，如果一个实体在文本中被提到了多次，使用了几种可能的描述和不同的名字形式，要求系统都要把它们识别出来，一个文本中的每个实体只有一个模板元素［Grishman and Sundheim, 1996］；
⑤ 模板关系（template relation, TR）：确定实体之间与特定领域无关的关系。
<!-- more -->

1999年起美国NIST组织了自动内容抽取（automatic content extraction, ACE）评测会议，旨在研究和
开发自动内容技术以支持对三种不同来源文本（普通文本、经语音识别后得到的文本、 由OCR识别得到的文本）的自动处理，以实现新闻语料中出现的实体、关系、事件等内容的自动抽取。评测任务设计:
实体检测与跟踪（entity detection and tracking, EDT）、数值检测与识别（value detection and recognition, VDR）、时间识别和规范化（time expression recognition and normalization, TERN）、关系检测与描述（relation detection and characterization, RDC）、事件检测与描述（event detection and characterization, EDC）和实体翻译（entity translation, ET）等。

### TF-IDF 关键词抽取
```python
import jieba.analyse
jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
```
`sentence` 为待提取的文本
`topK` 为返回几个 TF/IDF 权重最大的关键词，默认值为 `20`
`withWeight` 为是否一并返回关键词权重值，默认值为 `False`
`allowPOS` 仅包括指定词性的词，默认值为空，即不筛选. 如电商评论指定要形容词

关键词提取所使用逆向文件频率（IDF）文本语料库可以切换成自定义语料库的路径, 用法： `jieba.analyse.set_idf_path(file_name) # file_name为自定义语料库的路径`
自定义语料库示例见 https://github.com/fxsjy/jieba/blob/master/extra_dict/idf.txt.big
用法示例见 https://github.com/fxsjy/jieba/blob/master/test/extract_tags_idfpath.py

关键词提取所使用停止词（Stop Words）文本语料库可以切换成自定义语料库的路径, 用法： `jieba.analyse.set_stop_words(file_name) # file_name为自定义语料库的路径`
自定义语料库示例见 https://github.com/fxsjy/jieba/blob/master/extra_dict/stop_words.txt
用法示例见 https://github.com/fxsjy/jieba/blob/master/test/extract_tags_stop_words.py

关键词一并返回关键词权重值 https://github.com/fxsjy/jieba/blob/master/test/extract_tags_with_weight.py

### TextRank
论文：[TextRank: Bringing Order into Texts](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).
* 将待抽取关键词的文本进行分词
* 以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
* 计算图中节点的PageRank，是无向带权图
* 数据量越大，构建的图越精准

`jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))` 默认过滤词性。`jieba.analyse.TextRank()` 新建自定义实例

## 开放式信息抽取
处理的文本领域不再限定于规范的新闻文本或者某一领域文本，而是不限定领域的网络文本, 不仅需要考虑文本
特征，同时需要综合考虑网页结构特征和用户行为特征等。

### 开放式实体抽取
开放式实体抽取关注的是从海量、冗余、不规范的网络数据源上抽取出符合某个语义类的实体列表，侧重于抽取。

基于这样的假设：同类实体在网络上具有相似的网页结构或者相似的上下文特征。因此可以根据给出的特定语义类的若干实体（“种子”），找出该语义类包含的其他实体，其中特定语义类的标签可能是显式，也可能是隐式给出的。如给出“中国、美国、俄罗斯”这三个实体，要求找出“国家”这个语义类的其他实体诸如“德国、法国、日本”等。

训练步骤包含两部分：候选实体获取和候选实体置信度计算和排序。

具体训练过程：通常从种子实体出发，通过分析种子实体在语料中的上下文特征得到模板，根据模板得到更多候选实体，选取置信度高的候选实体作为新种子进行迭代，满足一定条件后停止迭代， 返回
历次置信度高的候选实体作为结果输出。

抽取比识别在任务上更加底层，实体抽取的结果可以作为列表支撑实体的识别。

对于中文而言，当不存在网页结构特征时，实体抽取任务变得更加困难，其中一个重要原因来自汉语分词，未知实体往往在分词过程中被分开。

### 实体消歧
Entity disambiguation的难点在于指称项多样性（name variation）和指称项歧义（name ambiguity）。
指称项多样性指一个实体概念可以用多种命名性指称项指称，如全称、别称、简称、拼写错误、多语言名称等。

单语言的实体消歧问题的主要方法：
1. 实体聚类消歧法：对每一个实体指称项抽取其上下文特征（包括词、实体等），并将其表示成特征向量；然后计算实体指称项之间的相似度；计算基于指称项之间的相似度时，可采用一定聚类算法将其聚类，将每个类看作一个实体概念。这种方法的核心任务是计算实体指称项之间的相似度，
  * 传统的方法是利用上下文的词信息建立词袋模型（bag-of-words, BOW）。
  * 针对人名消歧，采用基于图的算法，利用社会化关系的传递性考虑隐藏的实体关系知识。
  * 利用知识资源，如Wikipedia、Web上的链接信息、命名实体的同现信息、领域特定语料库等，来提升实体消歧的效果。
2. 实体链接消歧法：实体链接（entity linking）也称实体分辨或实体解析（entity resolution），或记录链接（record linkage）。基于实体链接消歧法的目的是解决基于聚类的实体消歧法不能显式地给出实体语义信息的问题，其基本任务是：给定一个实体指称项，将其链接到知识库中的实体概念上。实体链接的核心任务仍是计算实体指称项和候选实体之间的相似度，选择相似度最大的候选实体作为链接的目标实体。

实体消歧仍面临很多难题，包括空目标实体问题（NIL entity problem）（即实体知识库中不包含某指称项的目标实体）、知识库覆盖度有限、来自互联网的知识源可靠性差和知识库使用方法单一（集中于使用单文档特征）等。

### 开放式实体关系抽取
实体关系通常采用采用三元组表示：`(Arg1, Pred, Arg2)`， 其中，`Arg1`表示实体，`Arg2`表示实体关系值，通常也是实体，`Pred`表示关系名称，通常为动词、名词或者名词短语。
