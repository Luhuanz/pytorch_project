from transformers.configuration_bert import BertConfig
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertLayer, BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, _CONFIG_FOR_DOC
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

#这段代码导入了 transformers 包中的 add_code_sample_docstrings 和 add_start_docstrings_to_callable 函数。
# 为一个 Python 函数添加文档字符串，使得该函数能够在被调用时显示帮助信息。
# add_code_sample_docstrings 可以为函数添加示例代码文档，而 add_start_docstrings_to_callable 可以为函数添加描述文档。
# 这些文档字符串的格式符合 Google 风格的 Python Docstrings 标准，这种标准在 Python 社区中比较常用。

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_callable,
)


class WordEmbeddingAdapter(nn.Module):
    
    def __init__(self, config):
        super(WordEmbeddingAdapter, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(config.word_embed_dim, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        attn_W = torch.zeros(config.hidden_size, config.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, layer_output, word_embeddings, word_mask):
        """
        :param layer_output:bert layer的输出,[b_size, len_input, d_model]
        :param word_embeddings:每个汉字对应的词向量集合,[b_size, len_input, num_word, d_word]
        :param word_mask:每个汉字对应的词向量集合的attention mask, [b_size, len_input, num_word]
        """

        # transform
        # 将词向量，与字符向量进行维度对齐
# word_embeddings #torch.Size([4, 150, 3, 200])
        word_outputs = self.linear1(word_embeddings) #torch.Size([4, 150, 3, 768])
        word_outputs = self.tanh(word_outputs)
        word_outputs = self.linear2(word_outputs)
        word_outputs = self.dropout(word_outputs)   # word_outputs：[b_size, len_input, num_word, d_model]

        # 计算每个字符向量，与其对应的所有词向量的注意力权重，然后加权求和。采用双线性映射计算注意力权重
        # layer_output = layer_output.unsqueeze(2)    # layer_output：[b_size, len_input, 1, d_model]
        socres = torch.matmul(layer_output.unsqueeze(2), self.attn_W)  #torch.Size([4, 150, 1, 768])# [b_size, len_input, 1, d_model]
        socres = torch.matmul(socres, torch.transpose(word_outputs, 2, 3))  # [b_size, len_input, 1, num_word]
        socres = socres.squeeze(2)  # [b_size, len_input, num_word]
        socres.masked_fill_(word_mask.byte(), -1e9)  # 将pad的注意力设为很小的数
        socres = F.softmax(socres, dim=-1)  # [b_size, len_input, num_word]
        attn = socres.unsqueeze(-1)  # [b_size, len_input, num_word, 1]

        weighted_word_embedding = torch.sum(word_outputs * attn, dim=2)  # [N, L, D]   # 加权求和，得到每个汉字对应的词向量集合的表示
        layer_output = layer_output + weighted_word_embedding

        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output)

        return layer_output


class LEBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
#BertPooler是一个在BERT模型中用于从最后一层的输出计算整个序列表示的类。
# 在BERT模型的最后一层，有一个大小为（batch_size, seq_len, hidden_size）的输出张量，其中每个单词在隐藏空间中都有一个向量表示。
        self.pooler = BertPooler(config)

        self.init_weights()

#用于获取和设置输入嵌入层（input embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
#而 set_input_embeddings 函数用于更新这个 nn.Embedding 层的参数值。
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
#这是BERT模型的一个方法，用于剪枝模型中的注意力头。具体来说，
# 该方法接收一个字典heads_to_prune，字典的键是层数，值是要在该层中剪枝的注意力头的列表。
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        word_embeddings=None,
        word_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            word_embeddings=word_embeddings,
            word_mask=word_mask,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.word_embedding_adapter = WordEmbeddingAdapter(config)

    def forward(
        self,
        hidden_states,
        word_embeddings,
        word_mask,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # 在第i层之后，进行融合
            if i == self.config.add_layer:
                hidden_states = self.word_embedding_adapter(hidden_states, word_embeddings, word_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
