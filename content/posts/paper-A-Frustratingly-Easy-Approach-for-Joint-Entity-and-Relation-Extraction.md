title: A Frustratingly Easy Approach for Joint Entity and Relation Extraction
date: 2021-04-20
mathjax: true
categories:
- AI
- Paper
tags:
- NLP
- 2020
- NAACL
- Extraction

---

2020, NAACL 

data: ACE 04, ACE 05, SciERC

links: https://github.com/princeton-nlp/PURE

task: Entity and Relation Extraction

<!-- more -->

提出了一种简单但是有效的pipeline方法:builds on two independent pre-trained encoders and merely uses the entity model to provide input features for the relation model.

实验说明: validate the importance of 

- **learning distinct contextual representations for entities and relations**,
- fusing entity information at the input layer of the relation model,
- and incorporating global context.

从效果上看, 似乎是因为cross sentence的context加成更大

# 方法

Input: a sentence X consisting of n tokens `x1, . . . , xn`. Let `S = {s1, . . . , sm}` be all the possible spans in `X` of up to length `L` and `START(i)` and `END(i)` denote start and end indices of `si`.

![/images/papers/paper4.png](/images/papers/paper4.png)

## Named entity recognition

a standard span-based model following prior work (Lee et al., 2017; Luan et al., 2018, 2019; Wadden et al., 2019), 加入span-width embedding

![/images/papers/paper4-1.png](/images/papers/paper4-1.png)

## Relation model

1. observe that span he(si), he(sj) representations only capture contextual information around each individual entity and might fail to capture the dependencies between a specific pair of spans. 
2. We also hypothesize that sharing the contextual representations for different pairs of spans may be suboptimal.
3. instead processes each pair of spans independently and inserts typed markers at the input layer to highlight the subject and object and their types.

![/images/papers/paper4-2.png](/images/papers/paper4-2.png)

![/images/papers/paper4-3.png](/images/papers/paper4-3.png)

### Cross-sentence context

extending the sentence to a fixed window size `W` for both the entity and relation model. augment the input with `(W − n)/2` words from the left context and right context respectively (`W = 100` in our default model).

## Training & inference

![/images/papers/paper4-4.png](/images/papers/paper4-4.png)

![/images/papers/paper4-5.png](/images/papers/paper4-5.png)

## 3.3 Efficient Batch Computations

目标是避免多次encode一个句子

1. 把所有实体的 marker tokens 放在序列尾部, 
2. The position embeddings of the markers: 直接复用实体span的 start and end tokens
3. We enforce the text tokens to only attend to text tokens and not attend to the marker tokens while an entity marker token can attend to all the text tokens and all the 4 marker tokens associated with the same span pair.

# 效果

![/images/papers/paper4-6.png](/images/papers/paper4-6.png)