---
title: 语言模型
date: 2017-11-12
mathjax: true
author: "Cong Chan"
tags: ['NLP']
---
## 语言模型
语言模型Language modeling（LM）最初是针对语音识别问题而开发的, 现在广泛用于其他NLP应用中, 比如机器翻译需要利用LM来给翻译出的句子打分.
<!-- more -->
假设我们有一个语料库 - 某种语言的句子的无限集合$\mathcal{V^+}$（这些句子是由有限的词$\mathcal{V}$组成的）。例如，我们可能从网上获得大量文本。给定了此语料库，我们想估计LM的参数。这些参数包含语料库中所有单词的有限集合$\mathcal{V}$, 以及句子的概率分布函数$p(x_1, x_2, ..., x_n)$，必须满足
1. For any $\langle x_1...x_n \rangle \in \mathcal{V^+}$, $p(x_1, x_2, ..., x_n) ≥ 0$
2. $\sum_{\langle x_1...x_n \rangle \in \mathcal{V^+}}p(x_1, x_2, ..., x_n) = 1$

比如，当$\mathcal{V}$只有`cat, eat, fish`, 那么它组合成的句子按照人类的评价标准, 通顺程度从高到低是: `cat eat fish`, `fish eat cat`, `cat fish eat`, `eat cat fish`, `eat fish cat`, `fish cat eat`. 这些是可能出现的句子(还没出现的不代表未来不会出现), 从概率分布的角度看待, 这些句子的概率之和是`1`, 因为这三个词只能组成这几个句子. 而LM的意义就在于能够赋予`cat eat fish`最大的概率, 代替人来判断句子是否准确, 通俗的说是一个句子通顺打分机器.

广义的语言模型, 可以计算任何连续的单词或者任何其他序列数据（比如语音）出现的概率, 当然是以参数的训练样本的角度来看待。除了为每个词序列指定概率之外，语言模型还指定给定的单词（或单词序列）跟随前面的单词序列的似然概率。

语言模型本身即是一种概率模型. 概率模型是随机现象的数学表示，由样本空间，样本空间内的事件以及与每个事件相关的概率定义。目标是模拟一个事件发生的概率。

LM的任务就是为单词序列$w_{1:n}$分配概率$P(w_{1:n})$, 等同于给序列的每个位置预测可能出现的单词，给定前面的单词（作为条件），预测下一个单词出现的概率 `P(w|w1, w2, w3...)`。听起来有点像词性标注(Tagging)... 事实上最初为语言建模开发的参数估计技术也给词性标注做了不少贡献.

利用链式法则, $$P(w_{1:n}) = P(w_1)P(w_2|w_1)P(w_3|w_{1:2})P(w_4|w_{1:3})...P(w_n|w_{1:n-1}),$$ 最后一项基于`n-1`个词的条件概率计算难度非常大。为了简化LM参数的训练，利用`k`阶马尔可夫假设，声明序列的下一个词仅依赖于前`k`个词。如利用一阶马尔可夫假设得到`P(transparent | the water is so ) ≈ P(transparent | so)`.

使用马尔可夫假设简化前面的乘链:
$$\begin{align}
P(w_{1:n}) &= \prod_{i=1}^n P(w_i | w_1, ..., w_{i-1}) \\\\
&\propto \prod_{i=1}^n P(w_i | w_{i-k}, ..., w_{i-1}) \end{align}$$
在语料处理时，开头的句子前面需要相应的加上`k`个补丁符号`<s>`，才能计算第一个词的条件概率。LM也是一种生成模型, 一般是在句子末尾加上特殊符号`</s>`表示句子结束, 以方便生成任务时判断句子的生成结束.

固然Markov假设对于任意`k`阶都是有偏差的（毕竟句子可以有任意长的依赖性），但仍可以使用较小的k建模出较强的LM，并且几十年来一直是语言建模的主要方法。

对于LM参数中每一项似然概率的估算，可以使用**最大似然估计（MLE）**：$$P(w_{i}=m|w_{i-k:i-1}) = \frac{Count(w_{i-k:i})}{Count(w_{i-k:i-1})}$$

这个就是经典的N-gram模型。

## N-Gram语言模型
`N-Gram`语言模型是基于`N-1`阶马尔可夫假设且由MLE估算出的LM。`N-Gram`LM 预测下一个单词出现概率仅条件于前面的`(N-1)`个单词, 以`The students opened their books`为例:
* `Bi-gram`: 统计$P(w_{i}=m|w_{i-1})$, `P(students | the)`, `P(opened | students)`, ..., 属于`马尔可夫一阶模型`, 即当前`t`时间步的状态仅跟`t-1`相关.
* `Tri-gram`: `P(students | </s> The)`, `P(opened | The students)`, `马尔可夫二阶模型`
* `Four-gram`: 依此类推

特殊的`Uni-gram`: 统计$P(w_i)$, `P(the)`, `P(students)`, ..., 此时整个模型退化为词袋模型, 不再属于马尔可夫模型, 而是基于贝叶斯假设, 即各个单词是条件独立的. 所以一般`N-gram`是指`N>1`的.

N-Gram模型因为使用MLE估算参数，缺点很明显：
* 无法很好地解决NLP中的长距离依赖现象, 比如一般表现比较好的Trigram语言模型，没有考虑到两步之外的词
* 没有考虑词的相似性，泛化能力差。比如在训练集出现了`The cat is walking in the bedroom`,理论上应该泛化到给`A dog was running in a room`, 因为`dog`和`cat`(resp. “the” and “a”, “room” and “bedroom”, etc...)有类似的语义和语法定位.
* N-gram只是在测试语料库与训练语料库比较相似时表现才比较好。否则基于训练语料训练出来的参数肯定无法很好地评估测试语料，就像人无法对其不认识的语言做任何语法句法上的评价。
* 稀疏问题1：大多数高阶Gram几乎不会出现，虽然`u v w`在训练语料中从来没有出现过, 但我们不能简单地把`P(w | u, v)`定义为0，因为语言是千变万化的，有些词组虽然少见但不代表不存在。句子的概率是由各个gram似然概率相乘而来，如果仅仅因为一个词组出现次数为0就导致整个句子概率变为0, 那显然是不合理的.
* 稀疏问题2：部分低阶gram没有出现过，低阶gram的次数作为MLE公式中分母变为0，那计算就没法进行下去了.
* 一般而言，N越高，模型表现越好，但是更大的N使稀疏问题变得更糟。通常人们不会取大于5的N。
* 需要存储所有可能的N-Gram，所以模型的大小是 `O(exp(n))`, 需要大量的内存，而其实大部分都是出现次数为0.

### 平滑
针对数据稀疏问题（0概率的问题）, 可以使用各种平滑处理（Smoothing）.

加一（Laplace）平滑：最简单的平滑法，为所有事件（不管有没出现过）的频次加一，这样保证了没有0概率事件出现。这种平滑效果很差，因为齐夫定律`Zipf's law`的关系
> `Zipf's law`：在自然语言的语料库里，一个单词出现的频率与它在频率表里的排名成反比。

会有很多长尾单词很少甚至几乎没有出现过, 所以在总数为1的概率池子里, 为了给这些长尾单词分配至少频次1的概率, 需要从真正出现的单词(所谓真实发生的事件)中分走很多概率.

因此可以给Laplace平滑加入控制因子，变为 Add alpha smoothing。更多平滑方案参考[UoE-anlp](/UoE-anlp#平滑Smoothing)

## 语言模型评估方法
既然LM是用于评估句子是否准确的模型，那么在评价LM好坏时，就要看它在测试集上的表现如何。给定测试集包含$m$个句子$x^{(1)}, x^{(2)}, ..., x^{(m)}$, 各个句子的长度分别为$n_i$. LM给这些测试集句子评估的概率大小为$$\prod_{i=1}^m p(x^{(i)})$$ 这个数值越高，说明LM评估测试集句子的质量越好。**注意, 测试集必须是完全没有参与模型训练, 且是在人类标准中是好的句子.**

但在实际使用中, 我们往往使用上面这个概率的一种变换 - `困惑度`（`Perplexity`）来评价LM的质量. 首先取整个测试语料库的对数概率除以测试语料库中的单词总数$M$: $$l = \frac{1}{M} \log_2 \prod_{i=1}^m p(x^{(i)}) = \frac{1}{M} \sum_{i=1}^m \log_2 p(x^{(i)})$$
然后得到
$$\begin{align}
Perplexity &= 2^{-l} \\\\
&= 2^{-\frac{1}{M} \sum_{i=1}^m \log_2 p(x^{(i)})}\\\\
\\\\
&= t^{-1}
\end{align}$$
其中，$t = \sqrt[\leftroot{-2}\uproot{2}M]{\prod_{i=1}^m p(x^{(i)})}$, 作为测试集概率的几何平均. 例如，如果困惑等于100，则$t = 0.01$，表明几何平均值为0.01. 可以看到, Perplexity的值越小，语言模型建模测试集的能力就越好.

概率取对数转换可以避免数值下溢，可以把乘法转换为加法, 计算也更快.

困惑度为何就是一种好的衡量标准呢？对于任何一个任务，我们需要定义Baseline模型作为基准，如果后续有一个新的模型，但无法超过此baseline，那么我们认为这个新的模型是没有进步的。对于语言建模这一个任务，最无脑最简单的baseline，就是假设每一个位置的每个单词出现概率相等，这就是最大熵分布，即假设此baseline对这个任务一无所知，所有位置所有单词在它眼里都是没区别的(均匀分布)。如果词汇集(包含`</s>`)大小为`N`, 那么$$P_{i \in T}(w_i | w_{1:i-1}) = \frac{1}{N},$$ 此时的困惑度等于`N`, 即在均匀概率分布模型下，困惑度等于词汇量的大小。显而易见任何一个有效模型的困惑度必须小于类别个数. 此时困惑度可以理解为模型的**有效词汇量**：例如，词汇量大小为10,000, 而模型的困惑度为120，那么这大致说明有效的词汇量只有大概120个。最佳情况下，模型总是把测试集的概率预测为 1, 此时困惑度为 1。最坏情况下，概率预测为 0, 此时困惑度为正无穷。Baseline模型总是预测所有类别的概率都相同, 此时困惑度为词汇量大小（类别个数）。

目前很多神经网络框架计算语言模型的损失函数都是用交叉熵损失函数并取对数,
要得到perplexity，只需要把这个loss取指数运算。

那么困惑度一般都是多大呢？Goodman (“A bit of progress in language modeling”, figure 2) 评估了在英语数据上的unigram，bigram和trigram语言模型，词汇量为50,000。Goodman的报告结果显示，trigram模型的困惑度约为74，bigram模型为137，unigram模型为955。相比于Baseline模型困惑度50,000，trigram模型显然有了巨大的改进，且比bigram和unigram模型也有很大的改进。而更强大的SOTA神经语言模型，可以在wikitext-2数据集上跑出40以下的困惑度。

## 神经网络语言模型
神经网络模型解决了传统语言模型的一些缺点：它们允许越来越长的距离依赖，而参数数量仅线性增加，它们减少了手动设计backoff顺序的需要，并且它们支持跨不同上下文的泛化。

`Bengio et al. [2003]`提出的神经网络语言模型(NNLM, 确切的说是前馈神经网络语言模型), 把文本处理成n个k-gram词窗口$w_{i:i+k-1}$,  每个词转换为词镶嵌的形式$\mathcal{v}(w) \in \mathcal{R}^{d_w}$, 一整个窗口的词向量拼接为矩阵向量$x = [\mathcal{v}(w_0); ...; \mathcal{v}(w_{k-1})]$, 作为输入数据输入到一个1到2层的感知机.

训练数据的处理一般这么操作, 每个句子的开头加上`<s>`, 末尾加上`</s>`, 然后按照k大小的长度一段段截断成k-gram词窗口$w_{i:i+k-1}$. 每一段k-gram的词拼接为一个向量$x = (C(w_{i}), C(w_{i+1}), ···, C(w_{i+k-1}))$, 作为一个训练样本, 其末尾的下一个词$w_{i+k}$作为样本对应的预测标签$y_i = \mathcal{v}(w_{i+k})$. 训练时，以输出的词向量概率分布向量和对应正确标签的 one-hot-vector 间的 cross-entropy loss 为损失函数.

神经网络的参数数量比传统的N-gram少，因为其每增加一个词，参数就多$d_w$, 也就是线性增加, 而N-gram是多项式增加速率. 并且NNLM的参数矩阵对所有输入都是共享的, 这进一步减少了参数量. 虽然如此, NNLM的训练时间还是比N-gram LM长.

神经网络语言模型的泛化能力更好，因为相似的词具有相似的特征向量，并且因为概率函数（模型参数）是这些特征值的平滑函数，所以特征的微小变化相应地引起概率的微小变化。

真正影响NNLM计算效率的是输出层的softmax计算, 因为训练样本的词汇量$\mathcal{V}$往往很大. 输出层的softmax需要与隐含层参数矩阵$W^2 \in \mathcal{R}^{d_{hid} \times \mathcal{V}}$进行昂贵的矩阵向量乘法, 然后进行$\mathcal{V}$次对数操作. 这部分计算占据了大部分运行时间，使得大词汇量的NNLM建模令人望而却步。

后续发展的NNLM普遍使用循环神经网络（RNN, LSTM）来代替简单的前馈神经网络。循环神经网络可以理解为多层前馈神经网络叠加, 但各神经网络隐含层的参数是共享的. 句子逐词输入循环神经网络, 也就是循环神经网络使用同样参数方程来处理每一个词, 因此循环神经网络的参数量比前馈神经网络更少. 使用循环神经网络作为LM模型时, 同样最后一层还是使用softmax输出层。不同的是输入不再局限于定长的kgram词窗口，LSTM理论上可以接受无限长序列, 但事实上LSTM的记忆能力也是有限的, 太长就会遗忘掉前面的信息.

### 对大词汇量语言模型的尝试
Hierarchical softmax [Morin and Bengio, 2005]

Self-normalizing aproaches, 比如 noise-contrastive estimation (NCE) `[Mnih and Teh, 2012, Vaswani et al., 2013]` 或者在训练目标函数中加入正则化项 `[Devlin et al., 2014]`.

有关处理大输出词汇表的这些和其他技术的良好评论和比较，请参阅 `Chen et al. [2016]`.

## 参考资料
class notes by Michael Collins: http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf
Neural Network Methods in Natural Language Processing, by Yoav Goldberg

A Neural Probabilistic Language Model, Yoshua Bengio, 2003
