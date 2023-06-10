title: Best practice to define Reward Model with HuggingFace transformers
date: 2023-05-30
mathjax: true
categories:
- AI
tags:
- LLM
- Reward Modeling
- RLHF
---
[Cong Chen](https://congchan.github.io/)  
University of Edinburgh

*Started writing on May 25 2023*
*Released in May 30 2023*
---
There are various implementation of reward modeling in RLHF(reinforcement learning with human feedback), each has different pros and cons. Inspired by some open-sourced works about reward modeling, I would like to share one of the best practice for reward modeling. For those who are not familiar with reward modeling and RLHF, I recommend take a look at the Huggingface rlhf blog[^1] or OpenAI rlhf paper[^2]. 

<!-- more -->

# Suggested Practice
Reward modeling involves training a model to predict a scalar value as a reward for a given input text. This is achieved through a high-level architecture consisting of a deep transformers block and a value head with an output dimension of one. To streamline the training and inference stages, the `AutoModelForSequenceClassification` class can be utilized. One effective way to ensure consistency and compatibility is to use Huggingface transformers' self-defined model with `auto_map`. This involves defining different reward model backbones depending on the model type (such as Llama[^3] or Bloom[^4]), then using the `from_pretrained` method to load the model for either training or inference purposes.

Let us use a Llama pre-trained causal language model from HuggingFace decapoda-research/llama-7b-hf[^5] as an example and provide step-by-step instructions.

First, we would like to train this pre-trained causal language model on some reward datasets(such as the Anthropic/hh-rlhf Datasets[^6]). Let’s create a script called `modeling_llama_rw.py` in the pretrained model directory to let transformers know how to load the model.
```python
from transformers import LlamaPreTrainedModel, LlamaConfig, LlamaModel
import torch.nn as nn
import torch


class LlamaRewardModel(LlamaPreTrainedModel):
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).
    uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.model = LlamaModel(config)
        self.value_head = nn.Linear(config.hidden_size, self.num_labels)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def forward():
       ...

```

Next, we need to add some extra information to the config.json file by adding a new class called `LlamaRewardModel` in the `"architectures"` key and a new class mapping of `"auto_map": ["AutoModelForSequenceClassification": "modeling_llama_rm.LlamaRewardModel"]`.
```python
{
  "_name_or_path": "LLaMA-7B-Reward",
  "architectures": [
    "LlamaRewardModel"
  ],
  "auto_map": {
    "AutoModelForSequenceClassification": "modeling_llama_rm.LlamaRewardModel"
  },
  ...  # we could leave the rest untouched
}
```

With these changes, we can now load the model using the following line: `model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True)`. The `trust_remote_code=True` is required to make sure transformers API could call our customed class. This line can be used for both training and inference stages. After training the model, we can share it, along with the configuration and modeling files, for downstream use.

I have also provided an example of Bloom reward modeling in the my GitHub repository[^7].

# Some Details in Reward Modeling
There is no definitive method for calculating scalar rewards based on the output logits of input text. However, OpenAI has provided some reference implementations for us to learn from. They use the output logits of the final token of the input sentence, including paddings. This is similar to the Huggingface `AutoModelForSequenceClassification`. You can directly use this class.

My implementation adds more fine-grained details to the process of computing scalar rewards. Specifically, the logit of the last token before the sentence padding is retrieved to compute the reward. In cases where the input sentence is too long to contain padding, the last token of the truncated sentence is used instead. All of these details can be defined in the forward function of the modeling class. Additionally, aligning the model's pad_token_id with how the tokenizer preprocesses the input data is necessary for this implementation, which can be easily achieved by setting `model.config.pad_token_id = tokenizer.pad_token_id`.
```python
from transformers import LlamaPreTrainedModel, LlamaConfig, LlamaModel
import torch.nn as nn
import torch


class LlamaRewardModel(LlamaPreTrainedModel):
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).
    uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.model = LlamaModel(config)
        self.value_head = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            lm_labels=None,
            mc_labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        rewards = self.value_head(hidden_states).squeeze(-1)
        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.pad_token_id
        ends = input_ids.shape[1] - (input_ids == pad_token_id).type(torch.int64).sum(dim=1).view(-1, 1)
        ends = torch.clamp(ends - 1, min=0)
        rewards = torch.gather(rewards, 1, ends)
        return rewards

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()
```

# Introduction to Other Noteworthy Implementations
DeepSpeedChat's solution[^8][^9] involves building an `nn.module` model and utilizing Huggingface‘s Pre-trained Transformers blocks as `self.base_model`. However, in my opinion, this approach is not ideal as it lacks consistency between the training and inference stages. Nonetheless, their breakdown of the Huggingface API and torch API showcases a well-designed implementation.
```python
def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)

    if rlhf_training:
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"
        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'))

    return critic_model
```

This is how DeepSpeedChat define their reward_model: 
```python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

```






[^1]: https://huggingface.co/blog/rlhf
[^2]: Ouyang, Long, et al. Training Language Models to Follow Instructions with Human Feedback. arXiv:2203.02155, arXiv, 4 Mar. 2022. arXiv.org, http://arxiv.org/abs/2203.02155.
[^3]: https://ai.facebook.com/blog/large-language-model-llama-meta-ai/
[^4]: BigScience, BigScience Language Open-science Open-access Multilingual (BLOOM) Language Model. International, May 2021-May 2022
[^5]: https://huggingface.co/decapoda-research/llama-7b-hf
[^6]: https://huggingface.co/datasets/Anthropic/hh-rlhf
[^7]: https://github.com/congchan/rlhf_exps/tree/main/reward_modeling/test/bloomz-7b1
[^8]: [microsoft/DeepSpeedExamples/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L18](https://github.com/microsoft/DeepSpeedExamples/blob/95adffb17720b66d2888793851e4652ae28202ba/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L18)
[^9]: [microsoft/DeepSpeedExamples/applications/DeepSpeed-Chat/training/utils/model/reward_model.py#L11](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py#L11)
