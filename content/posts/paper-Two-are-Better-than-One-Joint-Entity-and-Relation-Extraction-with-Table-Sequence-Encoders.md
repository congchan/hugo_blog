title: Two are Better than One - Joint Entity and Relation Extraction with Table-Sequence Encoders
date: 2021-03-27
mathjax: true
categories:
- AI
- Paper
tags:
- NLP
- 2020
- EMNLP
- Extraction

---

2020, EMNLP

data: ACE 04, ACE 05, ADE, CoNLL04

links: https://github.com/LorrinWWW/two-are-better-than-one.

task: Entity and Relation Extraction

<!-- more -->


In this work, we propose the novel table-sequence encoders where two different encoders – a table encoder and a sequence encoder are designed to help each other in the representation learning process.

这篇ACL 2020文章认为, 之前的Joint learning方法侧重于learning a single encoder (usually learning representation in the form of a table) to capture information required for both tasks within the same space. We argue that it can be beneficial to design two distinct encoders to capture such two different types of information in the learning process.

1. First, these methods typically suffer from feature confusion as they use a single representation for the two tasks – NER and RE
2. Second, these methods underutilize the table structure as they usually convert it to a sequence and then use a sequence labeling approach to fill the table

# 方法

1. we focus on learning two types of representations, namely sequence representations and table representations, for NER and RE respectively.
2. we design a mechanism to allow them to interact with each other, in order to take advantage of the inherent association underlying the NER and RE tasks
3. use the attention weights of BERT for learning table representations.

Regard NER as a sequence labeling problem, where the gold entity tags yNER are in the standard BIO

![/images/papers/paper3.png](/images/papers/paper3.png)

![/images/papers/paper3-1.png](/images/papers/paper3-1.png)

## Model

The model consists of two types of interconnected encoders, a table encoder for table representation and a sequence encoder for sequence representation

![/images/papers/paper3-2.png](/images/papers/paper3-2.png)

- In each layer, the table encoder uses the sequence representation to construct the table representation;
- and then the sequence encoder uses the table representation to contextualize the sequence representation

### Table Encoder

1. first construct a non-contextualized table by concatenating every two vectors of the sequence representation followed by a fully-connected layer to halve the hidden size 

    ![/images/papers/paper3-3.png](/images/papers/paper3-3.png)

2. Next, we use the **Multi-Dimensional Recurrent Neural Networks (MD-RNN)** with Gated Recurrent Unit (GRU), iteratively compute the hidden states of each cell to form the contextualized table representation, to access the context from four directions for modeling 2D data

![/images/papers/paper3-4.png](/images/papers/paper3-4.png)

Empirically, we found the setting only considering cases (a) and (c) in Figure 4 achieves no worse performance than considering four cases altogether

![/images/papers/paper3-5.png](/images/papers/paper3-5.png)

![/images/papers/paper3-6.png](/images/papers/paper3-6.png)

### Sequence Encoder

we replace the scaled dot- product attention with our proposed table-guided attention.

![/images/papers/paper3-7.png](/images/papers/paper3-7.png)

## 4.4 Exploit Pre-trained Attention Weights

![/images/papers/paper3-8.png](/images/papers/paper3-8.png)

# 效果

![/images/papers/paper3-9.png](/images/papers/paper3-9.png)

![/images/papers/paper3-10.png](/images/papers/paper3-10.png)