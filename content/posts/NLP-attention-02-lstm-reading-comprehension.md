title: 机器阅读理解 - LSTM与注意力机制 - 斯坦福问答数据集 (SQuAD)
date: 2018-07-20
mathjax: true
categories:
- AI
- NLP
tags:
- Attention
- NLP
- TensorFlow
---
本文介绍注意力机制如何应用于阅读理解类任务, 并介绍了由此任务催生的一些注意力变种.
<!-- more -->

## 注意力机制应用于阅读理解
The Standford question and answer dataset [(SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) 是由 Rajpurkar 等人提出的一个较有挑战性的阅读理解数据集。该数据集包含 10 万个（问题，原文，答案）三元组，原文来自于 536 篇维基百科文章，而问题和答案的构建主要是通过众包的方式，让标注人员提出最多 5 个基于文章内容的问题并提供正确答案，且答案出现在原文中。SQuAD 和之前的完形填空类阅读理解数据集如 CNN/DM，CBT 等最大的区别在于：SQuAD 中的答案不在是单个实体或单词，而可能是一段短语，这使得其答案更难预测。SQuAD 包含公开的训练集和开发集，以及一个隐藏的测试集，其采用了与 ImageNet 类似的封闭评测的方式，研究人员需提交算法到一个开放平台，并由 SQuAD 官方人员进行测试并公布结果。

由于 SQuAD 的答案限定于来自原文，模型只需要判断原文中哪些词是答案即可，因此是一种抽取式的 QA 任务而不是生成式任务。简单的 SQuAD 的模型框架可以参考seq2seq：Embed 层，Encode 层 和 Decode 层。Embed 层负责将原文和问题中的 tokens 映射为向量表示；Encode 层主要使用 RNN 来对原文和问题进行编码，这样编码后每个 token 的向量表示就蕴含了上下文的语义信息；Decode 层则基于 query-aware 的原文表示来预测答案起始位置。

但这个文本数据集涉及问题，原文，答案三个部分, 特别是需要根据问题在原文中搜寻答案的范围, 这就涉及如果把问题的信息提取出来并作用于原文. 目前各种前沿模型的关注点几乎都是在如何捕捉问题和原文之间的交互关系，也就是在 Encode 层和 Decode 层之间, 使用一个 Interaction 层处理编码了问题语义信息的原文表示，即 query-aware 的原文表示，再输入给 Decode 层。而本来应用机器翻译Attention机制就能很好的处理这种交互。

虽然注意力机制大同小异，但是不同的注意力权重（打分函数）带来的效果是不一样的。比较常用的是就是使用[全局注意力机制](\attention#全局注意力机制)中提到的
$$
\begin{aligned}
    score_{general}(t' t) &= s^\top_{t'} W_\alpha h_t, \\\
\end{aligned}
$$
就是用一个交互矩阵$W_\alpha$来捕捉问题和原文之间的交互关系. 原文作者称之为 **Bilinear**.
```python
class Attention(object):
    def forwards_bilinear(self, hc, hq, hc_mask, hq_mask, max_context_length_placeholder,
                                max_question_length_placeholder, is_train, keep_prob):
        '''combine context hidden state(hc) and question hidden state(hq) with global attention
            bilinear score = hc.T *W *hq
        '''
        d_en = hc.get_shape().as_list()[-1]
        # (BS, MPL, MQL)
        interaction_weights = tf.get_variable("W_interaction", shape=[d_en, d_en])
        hc_W = tf.reshape(tf.reshape(hc, shape=[-1, d_en]) @ interaction_weights,
                          shape=[-1, max_context_length_placeholder, d_en])

        # (BS, MPL, HS * 2) @ (BS, HS * 2, MCL) -> (BS ,MCL, MQL)
        score = hc_W @ tf.transpose(hq, [0, 2, 1])
        # Create mask (BS, MPL) -> (BS, MPL, 1) -> (BS, MPL, MQL)
        hc_mask_aug = tf.tile(tf.expand_dims(hc_mask, -1), [1, 1, max_question_length_placeholder])
        hq_mask_aug = tf.tile(tf.expand_dims(hq_mask, -2), [1, max_context_length_placeholder, 1])
        hq_mask_aug = hc_mask_aug & hq_mask_aug
        score = softmax_mask_prepro(score, hq_mask_aug)

        # (BS, MPL, MQL)
        alignment_weights = tf.nn.softmax(score)

        # (BS, MPL, MQL) @ (BS, MQL, HS * 2) -> (BS, MPL, HS * 2)
        context_aware = tf.matmul(alignment_weights, hq)

        concat_hidden = tf.concat([context_aware, hc], axis=2)
        concat_hidden = tf.cond(is_train, lambda: tf.nn.dropout(concat_hidden, keep_prob), lambda: concat_hidden)

        # (HS * 4, HS * 2)
        Ws = tf.get_variable("Ws", shape=[d_en * 2, d_en])
        attention = tf.nn.tanh(tf.reshape(tf.reshape(concat_hidden, [-1, d_en * 2]) @ Ws,
                                          [-1, max_context_length_placeholder, d_en]))
        return (attention)

    def _similarity_matrix(self, hq, hc, max_question_length, max_context_length, question_mask, context_mask, is_train,
                           keep_prob):
        def _flatten(tensor, keep):
            fixed_shape = tensor.get_shape().as_list()
            start = len(fixed_shape) - keep

            # Calculate (BS * MCL * MQL)
            left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])

            # out_shape is simply HS * 2
            out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]

            # (BS * MCL * MQL, HS * 2)
            flat = tf.reshape(tensor, out_shape)
            return (flat)

        def _reconstruct(tensor, ref, keep):
            ref_shape = ref.get_shape().as_list()
            tensor_shape = tensor.get_shape().as_list()
            ref_stop = len(ref_shape) - keep
            tensor_start = len(tensor_shape) - keep

            # [BS, MCL, MQL]
            pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]

            # [1]
            keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
            # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
            # keep_shape = tensor.get_shape().as_list()[-keep:]

            # [BS, MCL, MQL, 1]
            target_shape = pre_shape + keep_shape
            out = tf.reshape(tensor, target_shape)
            out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
            return (out)

        # (BS, MCL, MQL, HS * 2)
        d = hq.get_shape().as_list()[-1]
        logging.debug("d is: {}".format(d))
        hc_aug = tf.tile(tf.reshape(hc, shape=[-1, max_context_length, 1, d]),
                         [1, 1, max_question_length, 1])

        # (BS, MCL, MQL, HS * 2)
        hq_aug = tf.tile(tf.reshape(hq, shape=[-1, 1, max_question_length, d]),
                         [1, max_context_length, 1, 1])

        # [(BS, MCL, MQL, HS * 2), (BS, MCL, MQL, HS * 2), (BS, MCL, MQL, HS * 2)]
        args = [hc_aug, hq_aug, hc_aug * hq_aug]

        # [(BS * MCL * MQL, HS * 2), (BS * MCL * MQL, HS * 2), (BS * MCL * MQL, HS * 2)]
        args_flat = [_flatten(arg, 1) for arg in args]
        args_flat = [tf.cond(is_train, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg) for arg in args_flat]

        d_concat = d * 3
        W = tf.get_variable("W", shape=[d_concat, 1])
        b = tf.get_variable("b", shape=[1])

        # Calculating a(h, u) = w_s^(t)[h; u; h * u]
        # (BS * MCL * MQL, HS * 6) @ (HS * 6, 1) + (1) -> (BS * MCL * MQL, 1)
        res = tf.concat(args_flat, 1) @ W + b

        # (BS * MCL * MQL, 1) -> (BS, MCL, MQL)
        similarity_matrix = _reconstruct(res, args[0], 1)
        logging.debug("similiarity_matrix after reconstruct: {}".format(similarity_matrix.get_shape()))
        context_mask_aug = tf.tile(tf.expand_dims(context_mask, 2), [1, 1, max_question_length])
        question_mask_aug = tf.tile(tf.expand_dims(question_mask, 1), [1, max_context_length, 1])

        mask_aug = context_mask_aug & question_mask_aug
        similarity_matrix = softmax_mask_prepro(similarity_matrix, mask_aug)

        return (similarity_matrix)

```

### Bi-Directional Attention Flow
[lSeo et al. (2016)](https://arxiv.org/abs/1611.01603)针对SQuAD提出了一个另一种更复杂的注意力机制, Bi-Directional Attention Flow (BiDAF)。
![](https://allenai.github.io/bi-att-flow/BiDAF.png "image from: https://allenai.github.io/bi-att-flow/")
BiDAF顾名思义那个就是问题与段落的双向的注意力机制, 分别是 Context-to-query (C2Q) attention 和 Query-to-context (Q2C) attention. 两者都是基于传统的段落的背景向量 $H$ 与问题的背景向量 $U$ 间相似矩阵(similarity matrix) $S \in \mathbb{R^{T×J}}$衍生出来的.
$$
S_{tj} = \alpha(H_{:t}, U_{:j}) \in R \\\
\alpha(h, u) = w^{\top}_{(S)}[h; u; h \odot u]
$$

> Where $S_{tj}$ indicates the similarity between  t-th context word and j-th query word, $\alpha$ is a trainable scalar function that encodes the similarity between its two input vectors, $H_{:t}$ is t-th column vector of H, and $U_{:j}$ is j-th column vector of U, $w_{(S)} \in  R^{6d}$ is a trainable weight vector, $[;]$ is vector concatenation across row.

相似矩阵S被用于计算两种方向的注意力向量.

> Context-to-query (C2Q) attention signifies which query words are most relevant to each context word

$$
\tilde{U_{:t}} = \sum_j \alpha_{tj} U_{:j} \\\
\alpha_t = softmax(S_{t:})
$$
其中 $\alpha_t \in R^J 表示$t$段落词对各个问题词的注意力权重

> Query-to-context (Q2C) attention signifies which context words have the closest similarity to one of the query words and are hence critical for answering the query.

对段落的注意力权重为:
$$
b = softmax(max_{col}(S)) \in R^T
$$
其中$max_{col}$是在每行选出最大值.
然后对段落背景向量进行注意力加权:
$$
\tilde{h} = \sum_t b_t  H_{:t} \in R^{2d}
$$
这个$\tilde{h}$向量指的是在query眼里最重要的段落次的加权求和. 因为$\tilde{h}$是在每一个内去最大值, 所以还需要从新把$\tilde{h}$的值在每一个铺开$T$次得到一个$\tilde{H} \in R^{2dxT}$向量以方便后续的计算.

最后, 段落的embeddings向量和注意力向量结合为$G$, $G$的每一列向量可以理解为每个段落词的 query-aware representation:
$$
G_{:t} = \beta(H_{:t}, \tilde{U_{:t}}, \tilde{H_{:t}}) \in R^{d_G}
$$
> where $G_{:t}$ is the t-th column vector (corresponding to t-th context word), β is a trainable vector function that fuses its (three) input vectors, and $d_G$ is the output dimension of the β function.

β 函数可以是任意的神经网络, 但是文章中指出使用简单的函数如 $\beta(h, \tilde{u}, \tilde{h}) = [h; \tilde{u}; h \odot \tilde{u}; h \odot \tilde{h}] \in R^{8dxT}$ (i.e., dG = 8d) 表现已经很好了。

```python
class Attention(object):
    def forwards_complex(self, hc, hq, hc_mask, hq_mask, max_context_length_placeholder,
                  max_question_length_placeholder, is_train, keep_prob):
       '''combine context hidden state(hc) and question hidden state(hq) with attention
            measured similarity = hc : hq : hc.T * hq
       '''
       s = self._similarity_matrix(hq, hc, max_question_length_placeholder,
       max_context_length_placeholder, hq_mask, hc_mask, is_train, keep_prob)
       # C2Q

       # (BS, MCL, MQL)
       weights_c2q = tf.nn.softmax(s)

       # (BS, MCL, MQL) @ (BS, MQL, HS * 2) -> (BS, MCL, HS * 2)
       query_aware = weights_c2q @ hq

       # Q2C

       # (BS, MCL, MQL) -> (BS, MCL)
       # We are effectively looking through all the question words j's to some context word i and finding the
       # maximum of those context words
       score_q2c = tf.reduce_max(s, axis=-1)

       # (BS, MCL)
       weights_q2c = tf.expand_dims(tf.nn.softmax(score_q2c), -1)

       # (BS, HS)
       context_aware = tf.reduce_sum(tf.multiply(weights_q2c, hc), axis=1)

       # (BS, MCL, HS * 2)
       context_aware = tf.tile(tf.expand_dims(context_aware, 1), [1, max_context_length_placeholder, 1])

       # [(BS, MCL, HS * 2), (BS, MCL, HS * 2), (BS, MCL, HS * 2), (BS, MCL, HS * 2)]
       biattention = tf.nn.tanh(tf.concat([hc, query_aware, hc * query_aware, hc * context_aware], 2))

       return (biattention)

    def _similarity_matrix(self, hq, hc, max_question_length, max_context_length, question_mask, context_mask, is_train,
                          keep_prob):
       def _flatten(tensor, keep):
           fixed_shape = tensor.get_shape().as_list()
           start = len(fixed_shape) - keep

           # Calculate (BS * MCL * MQL)
           left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])

           # out_shape is simply HS * 2
           out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]

           # (BS * MCL * MQL, HS * 2)
           flat = tf.reshape(tensor, out_shape)
           return (flat)

       def _reconstruct(tensor, ref, keep):
           ref_shape = ref.get_shape().as_list()
           tensor_shape = tensor.get_shape().as_list()
           ref_stop = len(ref_shape) - keep
           tensor_start = len(tensor_shape) - keep

           # [BS, MCL, MQL]
           pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]

           # [1]
           keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
           # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
           # keep_shape = tensor.get_shape().as_list()[-keep:]

           # [BS, MCL, MQL, 1]
           target_shape = pre_shape + keep_shape
           out = tf.reshape(tensor, target_shape)
           out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
           return (out)

       # (BS, MCL, MQL, HS * 2)
       d = hq.get_shape().as_list()[-1]
       logging.debug("d is: {}".format(d))
       hc_aug = tf.tile(tf.reshape(hc, shape=[-1, max_context_length, 1, d]),
                        [1, 1, max_question_length, 1])

       # (BS, MCL, MQL, HS * 2)
       hq_aug = tf.tile(tf.reshape(hq, shape=[-1, 1, max_question_length, d]),
                        [1, max_context_length, 1, 1])

       # [(BS, MCL, MQL, HS * 2), (BS, MCL, MQL, HS * 2), (BS, MCL, MQL, HS * 2)]
       args = [hc_aug, hq_aug, hc_aug * hq_aug]

       # [(BS * MCL * MQL, HS * 2), (BS * MCL * MQL, HS * 2), (BS * MCL * MQL, HS * 2)]
       args_flat = [_flatten(arg, 1) for arg in args]
       args_flat = [tf.cond(is_train, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg) for arg in args_flat]

       d_concat = d * 3
       W = tf.get_variable("W", shape=[d_concat, 1])
       b = tf.get_variable("b", shape=[1])

       # Calculating a(h, u) = w_s^(t)[h; u; h * u]
       # (BS * MCL * MQL, HS * 6) @ (HS * 6, 1) + (1) -> (BS * MCL * MQL, 1)
       res = tf.concat(args_flat, 1) @ W + b

       # (BS * MCL * MQL, 1) -> (BS, MCL, MQL)
       similarity_matrix = _reconstruct(res, args[0], 1)
       logging.debug("similiarity_matrix after reconstruct: {}".format(similarity_matrix.get_shape()))
       context_mask_aug = tf.tile(tf.expand_dims(context_mask, 2), [1, 1, max_question_length])
       question_mask_aug = tf.tile(tf.expand_dims(question_mask, 1), [1, max_context_length, 1])

       mask_aug = context_mask_aug & question_mask_aug
       similarity_matrix = softmax_mask_prepro(similarity_matrix, mask_aug)

       return (similarity_matrix)
```

## 数据处理
内容段落摘自维基百科文章中的536篇文章，包含107,785对问题和答案，这使得SQuAD显着大于以前任何人类标注的数据集。在该数据集中，80％的数据用于训练，10％用于验证, 剩余10％用于测试。在训练集中，进一步划分出5％用于训练时的验证。

与其他问答数据集相比，SQUAD具有比较独特的特征，所有答案都是出自相应的上下文中。对于每一个段落, 众包人员生成几个问题，并选择原段落中的一小段作为答案. 答案由两个index组成, 对应答案在段落中的起始位置。因此，SQuAD数据集的答案可能比其他以单个单词和实体为答案为主的数据集长得多。实例:
> Question:
Why was Tesla returned to Gospic?

> Context paragraph:
On 24 March 1879, Tesla was returned to Gospicunder police guard for **not having a residence permit**...

> Answer:
{12, 16}

### Embedding
词向量使用预训练好的 Glove embedding.
>Glove is a log-bilinear regression model that combines the advantages of global matrix factorization and local context window methods.

```Python
def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

embeddings = load_glove_embeddings(embed_path)
```

```Python
class Model(metaclass=ABCMeta):
    ...
    @abstractmethod
    def setup_embeddings(self):
        pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return: embeddings representaion of question and context.
        """
        with tf.variable_scope("embeddings"):
            if self.config.RE_TRAIN_EMBED:
                embeddings = tf.get_variable("embeddings", initializer=self.embeddings)
            else:
                embeddings = tf.cast(self.embeddings, dtype=tf.float32)

            question_embeddings = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
            question_embeddings = tf.reshape(question_embeddings,
                        shape = [-1, self.max_question_length_placeholder, self.config.embedding_size])

            context_embeddings = tf.nn.embedding_lookup(embeddings, self.context_placeholder)
            context_embeddings = tf.reshape(context_embeddings,
                        shape = [-1, self.max_context_length_placeholder, self.config.embedding_size])

        return question_embeddings, context_embeddings
```

## 模型
整体的模型由Embedding层，Encodr层，Attention层，Decoder层组成

### Encoder
编码器就是一个双向GRU层:
```python
class Encoder(object):
    """
    In a generalized encode function, you pass in your inputs,
    masks, and an initial hidden state input into this function.
    :param inputs: Symbolic representations of your input
    :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                  through masked steps
    :param encoder_state_input: (Optional) pass this as initial hidden state
                                to tf.nn.dynamic_rnn to build conditional representations
    :return:
            outputs: The RNN output Tensor
                      an encoded representation of your input.
                      It can be context-level representation,
                      word-level representation, or both.
            state: The final state.
    """
    def __init__(self, state_size):
        self.state_size = state_size

    def encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, reuse=False, keep_prob = 1.0):
        return BiGRU_layer(inputs, masks, self.state_size, initial_state_fw, initial_state_bw, reuse, keep_prob)
```

```python
def BiGRU_layer(inputs, masks, state_size, initial_state_fw=None, initial_state_bw=None, reuse = False, keep_prob=1.0):
        ''' Wrapped BiGRU_layer for reuse'''
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        cell_fw = tf.contrib.rnn.GRUCell(state_size, reuse = reuse)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.GRUCell(state_size, reuse = reuse)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

        sequence_length = tf.reduce_sum(tf.cast(masks, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])

        # Outputs Tensor shaped: [batch_size, max_time, cell.output_size]
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                            cell_fw = cell_fw,\
                                            cell_bw = cell_bw,\
                                            inputs = inputs,\
                                            sequence_length = sequence_length,
                                            initial_state_fw = initial_state_fw,\
                                            initial_state_bw = initial_state_bw,
                                            dtype = tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], 2)
        return outputs, final_state_fw, final_state_bw
```

### Decoder
解码器也包含一个双向GRU层，输出的状态分别由两个softmax分类器计算出预测的答案的 start 和 end index 位置:
```python
class Decoder(object):
    """
    takes in a knowledge representation
    and output a probability estimation over
    all paragraph tokens on which token should be
    the start of the answer span, and which should be
    the end of the answer span.
    :param knowledge_rep: it is a representation of the paragraph and question,
                          decided by how you choose to implement the encoder
    :return: (start, end)
    """
    def __init__(self, output_size, state_size=None):
        self.output_size = output_size
        self.state_size = state_size

    def decode(self, knowledge_rep, mask, max_input_length, keep_prob = 1.0):
        '''Decode with BiGRU'''
        with tf.variable_scope('Modeling'):
            outputs, _, _ = BiGRU_layer(knowledge_rep, mask, self.state_size, keep_prob=keep_prob)

        with tf.variable_scope("start"):
            start = self.get_logit(outputs, max_input_length)
            start = softmax_mask_prepro(start, mask)

        with tf.variable_scope("end"):
            end = self.get_logit(outputs, max_input_length)
            end = softmax_mask_prepro(end, mask)

        return (start, end)


    def get_logit(self, inputs, max_inputs_length):
        ''' Get the logit (-inf, inf). '''
        d = inputs.get_shape().as_list()[-1]
        assert inputs.get_shape().ndims == 3, ("Got {}".format(inputs.get_shape().ndims))
        inputs = tf.reshape(inputs, shape = [-1, d])
        W = tf.get_variable('W', initializer=tf.contrib.layers.xavier_initializer(),
                             shape=(d, 1), dtype=tf.float32)
        pred = tf.matmul(inputs, W)
        pred = tf.reshape(pred, shape = [-1, max_inputs_length])
        tf.summary.histogram('logit', pred)
        return pred
```

### 搭建整个系统
在整个QASystem类中初始化这些功能层:
```python
class QASystem(Model):
    def __init__(self, embeddings, config):
        """ Initializes System """
        self.embeddings = embeddings
        self.config = config

        self.encoder = Encoder(config.encoder_state_size)
        self.decoder = Decoder(output_size=config.output_size, state_size = config.decoder_state_size)
        self.attention = Attention()

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))

        self.answer_start_placeholder = tf.placeholder(tf.int32)
        self.answer_end_placeholder = tf.placeholder(tf.int32)

        self.max_context_length_placeholder = tf.placeholder(tf.int32)
        self.max_question_length_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

        # ==== assemble pieces ====
        with tf.variable_scope(self.config.which_model, initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embeddings, self.context_embeddings = self.setup_embeddings()
            self.preds = self.setup_system()
            self.loss = self.setup_loss(self.preds)
            self.f1_train = tf.Variable(0., tf.float64)
            self.EM_train = tf.Variable(0., tf.float64)
            self.f1_val = tf.Variable(0., tf.float64)
            self.EM_val = tf.Variable(0., tf.float64)
            tf.summary.scalar('f1_train', self.f1_train)
            tf.summary.scalar('EM_train', self.EM_train)
            tf.summary.scalar('f1_val', self.f1_val)
            tf.summary.scalar('EM_val', self.EM_val)

        # ==== set up training/updating procedure ====
        ''' With gradient clipping'''
        opt_op = get_optimizer(self.config.optimizer, self.loss, config.max_gradient_norm, config.learning_rate)

        if config.exdma_weight_decay is not None:
            self.train_op = self.build_exdma(opt_op)
        else:
            self.train_op = opt_op
        self.merged = tf.summary.merge_all()
```

把各个功能层搭建成一个完整的模型:
```python
def setup_system(self):
    """
    Connect all parts of your system here:
    After your modularized implementation of encoder and decoder
    you should call various functions inside encoder, decoder here
    to assemble your reading comprehension system!
    context: [None, max_context_length, d]
    question: [None, max_question_length, d]
    :return:
    """
    d = self.context_embeddings.get_shape().as_list()[-1]

    '''Step 1: encode context and question, respectively, with independent weights
    e.g. hq = encode_question(question)  # get U (d*J) as representation of q
    e.g. hc = encode_context(context, q_state)   # get H (d*T) as representation of x
    '''

    with tf.variable_scope('question'):
        hq, question_state_fw, question_state_bw = \
            self.encoder.BiGRU_encode(self.question_embeddings, self.question_mask_placeholder,
                                keep_prob = self.dropout_placeholder)
        if self.config.QA_ENCODER_SHARE:
            hc, context_state_fw, context_state_bw =\
                 self.encoder.BiGRU_encode(self.context_embeddings, self.context_mask_placeholder,
                         initial_state_fw = question_state_fw, initial_state_bw = question_state_bw,
                         reuse = True, keep_prob = self.dropout_placeholder)

    if not self.config.QA_ENCODER_SHARE:
        with tf.variable_scope('context'):
            hc, context_state_fw, context_state_bw =\
                 self.encoder.BiGRU_encode(self.context_embeddings, self.context_mask_placeholder,
                         initial_state_fw = question_state_fw, initial_state_bw = question_state_bw,
                                     keep_prob=self.dropout_placeholder)

    d_Bi = self.config.encoder_state_size*2
    assert hc.get_shape().as_list() == [None, None, d_Bi], (
            "Expected {}, got {}".format([None, self.max_context_length_placeholder,
            self.config.encoder_state_size], hc.get_shape().as_list()))
    assert hq.get_shape().as_list() == [None, None, d_Bi], (
            "Expected {}, got {}".format([None, self.max_question_length_placeholder,
            self.config.encoder_state_size], hq.get_shape().as_list()))

    '''Step 2: combine context hidden state(hc) and question hidden state(hq) with attention
         measured similarity = hc.T * hq
         Context-to-query (C2Q) attention signifies which query words are most relevant to each P context word.
            attention_c2q = softmax(similarity)
            hq_hat = sum(attention_c2q*hq)
         Query-to-context (Q2C) attention signifies which context words have the closest similarity
            to one of the query words and are hence critical for answering the query.
            attention_q2c = softmax(similarity.T)
            hc_hat = sum(attention_q2c*hc)
         combine with β activation: β function can be an arbitrary trainable neural network
         g = β(hc, hq, hc_hat, hq_hat)
    '''
    # concat[h, u_a, h*u_a, h*h_a]
    attention = self.attention.forwards_bilinear(hc, hq, self.context_mask_placeholder, self.question_mask_placeholder,
                                max_context_length_placeholder = self.max_context_length_placeholder,
                                max_question_length_placeholder = self.max_question_length_placeholder,
                                is_train=(self.dropout_placeholder < 1.0), keep_prob=self.dropout_placeholder)
    d_com = d_Bi*4



    '''Step 3: decoding   '''
    with tf.variable_scope("decoding"):
        start, end = self.decoder.BiGRU_decode(attention, self.context_mask_placeholder,
                                self.max_context_length_placeholder, self.dropout_placeholder)
    return start, end
```
