---
title: 神经网络用于文本分类
date: 2018-01-15
mathjax: true
author: "Cong Chan"
tags: ['NLP']
---
## 文本分类
文本分类是很多业务问题中广泛使用到的NLP/监督机器学习（ML）。文本分类的目标是自动将文本/文档分类为一个或多个预定义类别。目前的成熟思路是用词向量解码文本，然后使用传统机器学习模型或者深度神经网络模型来做分类。

文本分类是学术界和工业界非常活跃的研究领域。本文主要介绍用于文本分类的几种神经网络模型方法，并比较它们的性能，代码实现主要基于Keras。文中代码都在这个[DeepText](https://github.com/congchan/DeepText)GitHub项目中.
<!-- more -->

文本分类的一些示例包括：
1. 从社交媒体中了解受众情绪（😁 😐 😥）
2. 检测垃圾邮件和非垃圾邮件
3. 自动标记客户查询
4. 将新闻文章📰分类为预定义主题

## 端到端文本分类流水线
端到端文本分类流水线由以下组件组成：
1. 训练文本：输入文本，有监督模型能够通过已标注数据来学习和预测所需的类。
2. 特征向量：特征向量是用于解码输入数据特征的信息的向量。
3. 标签：预定义的类别/类，作为模型预测的目标。
4. 算法模型：能够处理文本分类的算法（在我们的例子中：CNN，RNN，HAN, Fasttext）
5. 预测：已经在历史数据集上训练过的模型，可以用于执行标签预测。

这里使用汽车消费者的评测数据集，在`tsv`文件中, 第一列是序号对我们没用, 第二列是`label(0, 1)`，分别代表`（消极，积极）`评价，第三列是文本.
```
1	操控性舒服、油耗低，性价比高
0	动力的确有点点让我相信了up的确是个代步车而已!
1	1。车的外观很喜欢。2。省油，现在磨合期7.3，相信以后还会下降。
1	内饰的做工和用料同级别同价位最厚道的
0	减震系统太硬！
```
数据处理使用的类，具体见[代码链接](https://github.com/congchan/DeepText/blob/a33fe1b8e895916b26bc658f0a02ac8253291d8a/data_process.py#L29)
```python
class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets.
        helper funcitons [read_tsv, read_text, read_json]
    """
    ...

class SampleProcessor(DataProcessor):
    """ Sample processor for the classification data set.
        Tranform the text to tensor for training
        if use pre-train model, need vocabulary file
        usage:
            process data files
            >>> processer = SampleProcessor(config, )

            provide your own data in list format [train_X, train_Y, test_X, test_Y]
            >>> processer = SampleProcessor(config, data)

    """
```

### 词向量
使用包含外部知识的embedding表达字词是目前的主流方法，经典的如word2vec，GLoVe，较新进的 ELMo，BERT，等预训练向量，集成了关于单词的新信息（词汇和语义），这些信息已经在非常大的数据集上进行了训练和提炼。

在这里的模型，都允许我们直接载入外部的 embedding 参数。

特别是提供了通过预训练的BERT获取中文单词的向量表达的接口. 最好是使用在自己文本上fine-tune过的预训练BERT模型.
```python
@staticmethod
def load_bert_embedding(vob_size, emb_size, word2id):
    """ Get bert pre-trained representation,
        for example, pre-trained chinese_L-12_H-768_A-12,
            the hidden_size is 768
    """
    ...
    return rep_matrix
```
输入你的词汇表, 返回各个词汇对应的向量, 以词典形式返回. 内部的工作机制是把每一个单词都用拼接起来, 之间用BERT的句子分隔符`[SEP]`隔开. 在返回的token level 的向量中重新pool出各个词汇的表达. 这个方法具体的效果有待验证.

### Fasttext文本分类
Fasttext 非常适合处理一些显而易见，不需要推理，情况比较单纯的文本分类问题。它就是一个词袋模型，把文本所有单词的向量pool在一起，得出整个文本的向量表达，这个文本向量使用softmax分类器得出不同标签的概率分布。为了捕捉词之间的顺序，fasttext加入了ngram特征。详细推荐看这两篇文章
1. Enriching Word Vectors with Subword Information, P. Bojanowski, E. Grave, A. Joulin, T. Mikolov
2. Bag of Tricks for Efficient Text Classification, A. Joulin, E. Grave, P. Bojanowski, T. Mikolov
![](/images/fasttext.png "Image taken from the original paper")
[代码链接](https://github.com/congchan/DeepText/blob/a33fe1b8e895916b26bc658f0a02ac8253291d8a/models.py#L133)
```python
def fasttext(max_length, emb_size, max_words, class_num, pre_train_emb=None):
  """ return single label classification fasttext model
      paper: Bag of Tricks for Efficient Text Classification
      The original paper use average pooling.
      In many Kaggle application, Max Pooling is found to be useful
  """
  input = Input(shape=(max_length,), dtype='int32', name='input')

  embeddings_initializer = 'uniform'
  if pre_train_emb is not None:
      embeddings_initializer = initializers.Constant(pre_train_emb)
  embed_input = Embedding(output_dim=emb_size, dtype='float32', input_dim=max_words + 1,
                          input_length=max_length,
                          embeddings_initializer=embeddings_initializer,
                          trainable=True
                          )(input)

  drop_out_input = Dropout(0.5, name='dropout_layer')(embed_input)
  ave_pool = GlobalAveragePooling1D()(drop_out_input)
  max_pool = GlobalMaxPooling1D()(drop_out_input)
  concat_pool = concatenate([ave_pool, max_pool])
  output = Dense(class_num, activation='softmax', name='output')(concat_pool)
  model = Model(inputs=[input], outputs=output)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
```
对于中文文本，如果数据集不是很干净的话（比如有错别字），考虑使用特殊超参的fasttext。一般来说fasttext在英文中的`char+ngram`的窗口大小一般取值`3 ~ 6`，但是在处理中文时，为了去除输入中的噪声，那么可以把这个窗口限制为`1~2`，因为小窗口有利于模型去捕获**错别字**（错误词一般都是其中的一个字表达成同音异形的另一个字），比如小ngram窗口fasttext学出来的`"似乎"`近似词很有可能包含`"是乎"`等内部包含错别字的词，这样等于让fasttext拥有了识别错别字的词的能力。

### 卷积神经网络（CNN）文本分类
CNN通常用于计算机视觉，但它们最近已应用于各种NLP任务，结果很有前景。

简要地说，在文本数据上使用CNN时，当检测到特殊的 pattern，每个卷积的结果都将触发。通过改变内核的大小并连接它们的输出，你可以自己检测多个大小的模式（2, 3或5个相邻的单词）。Patterns 可以是表达式（如 ngrams），因此CNN可以在句子中识别它们而不管它们的位置如何。
![](/images/textCNN.png "Image Reference : Understanding convolutional neural networks for nlp")

参数使用 128 个 filters，大小从1到4。模型架构如图![](/images/textCNNarch.png "reference from https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f")
[代码链接](https://github.com/congchan/DeepText/blob/a33fe1b8e895916b26bc658f0a02ac8253291d8a/models.py#L161)
```python
def text_cnn(max_length, emb_size, max_words, class_num, pre_train_emb=None):
  " textCNN model "
  ...
  cnn1_1    = Conv1D(128, 1, padding='same', strides=1)(drop_out_layer)
  ...
  cnn1      = GlobalMaxPooling1D()(cnn1_2_at)

  cnn2_1    = Conv1D(128, 2, padding='same', strides=1)(drop_out_layer)
  ...
  cnn2      = GlobalMaxPooling1D()(cnn2_2_at)

  cnn3_1    = Conv1D(128, 4, padding='same', strides=1)(drop_out_layer)
  ...
  cnn3      = GlobalMaxPooling1D()(cnn3_2_at)

  concat_cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
  ...
  return model
```
用于text的CNN不仅更容易并行化运算，而且很容易成为一个数据集上的很强的baseline（除非这个分类任务很难）。根据数据的情况选择模型，如果ngram特征很重要，使用textCNN，如果文本长距离依赖比较明显，考虑使用RNN。

### RNN用于文本分类
RNN用于文本分类的话，seq2one 架构，把不定长序列解码为定长向量，再把这个输出向量用softmax函数计算出各标签的概率分布。RNN(LSTM/GRU)因为处理长文本的能力较弱，目前一般需要加上注意力机制。这里暂时简单粗暴的用双向GRU来定义核心的encoder.
```python
def text_rnn(max_length, emb_size, max_words, class_num, pre_train_emb=None):
    " Text RNN model using GRU cell"
    return _bilstm_attention(max_length, emb_size, max_words, class_num, False, pre_train_emb)

def text_rnn_attention(max_length, emb_size, max_words, class_num, pre_train_emb=None):
  " Text RNN model using GRU cell with attention mechanism"
  return _bilstm_attention(max_length, emb_size, max_words, class_num, True, pre_train_emb)
```

#### RCNN

### Hierarchical Attention Network (HAN)

## Reference
Enriching Word Vectors with Subword Information, P. Bojanowski, E. Grave, A. Joulin, T. Mikolov
Bag of Tricks for Efficient Text Classification, A. Joulin, E. Grave, P. Bojanowski, T. Mikolov
https://arxiv.org/abs/1408.5882 Yoon Kim
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/ by Denny Britz.
Understanding convolutional neural networks for nlp
https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f
[Hierarchical Attention Networks for Document Classification ](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
