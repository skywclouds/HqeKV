# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import fast_hadamard_transform
from model.public_func import (get_block_token_idx, make_sum_equal, last_attn_weights)

from quant.hqe import KV_cache_hqe
from quant.hqe_2 import KV_cache_hqe_2
from quant.hqe_matmul import qkv_matmul_hqe, qkv_matmul_hqe_2

import time
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, CausalLMOutputWithPast,)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import (
    auto_docstring, can_return_tuple, LossKwargs, add_start_docstrings,
    add_start_docstrings_to_model_forward, logging,
    replace_return_docstrings)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding, 
    LlamaPreTrainedModel, apply_rotary_pos_emb, repeat_kv)

logger = logging.get_logger(__name__)

class LlamaAttention_hqe(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.quant_strategy = config.quant_strategy
        self.times_range = config.times_range
        self.ratio_e_0 = torch.tensor([config.bit_0], dtype=torch.float32)
        self.ratio_e = self.ratio_e_0.clone()
        self.ratio_q = torch.tensor([config.bit_4, config.bit_2, config.bit_1], dtype=torch.float32)
        self.flash_length = 0
        self.token_num = 0
        self.compress_time = 128 * [0.0]
        self.decoding_time = 128 * [0.0]
        self.compress_length = 0
        self.current_length = 0
        self.history_reserve_length = 0
        self.last_weights = last_attn_weights(last_length=20)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
        auxiliary_parameters: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # 更新未量化的长度
        self.current_length += kv_seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        last_token_num = 20
        quant_strategy = self.quant_strategy
        # 这个是decoding阶段, else部分是prefilling阶段
        if past_key_value is not None:
            # start_time = time.time()
            # 把新生成的 K 和 V 拼接到以往的 KV 中
            new_K = torch.cat((past_key_value[-3], key_states), dim=2) if past_key_value[-3] is not None else key_states
            new_V = torch.cat((past_key_value[-2], value_states), dim=2) if past_key_value[-2]is not None else value_states
            past_key_value = past_key_value[:-3] + (new_K, new_V, kv_seq_len)
            # 计算attention
            start_time = time.time()
            attn_weights, attn_output = qkv_matmul_hqe_2(query_states, past_key_value, attention_mask, self.head_dim, quant_strategy)
            end_time = time.time()
            
            # self.decoding_time[self.token_num] = (end_time - start_time)
        # 这个是prefilling阶段
        else:
            start_time = time.time()
            if self.config._attn_implementation == 'flash_attention_2' and kv_seq_len > last_token_num:
                self.flash_length = kv_seq_len - last_token_num
                query_states_last = query_states[:, :, -last_token_num:, :]
                attn_output, _ = flash_attention_forward(
                    self, 
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    dropout = 0,
                    scaling=self.scaling,
                    **kwargs)
                # attn_output = attn_output.transpose(1, 2).contiguous()
                n_rep = self.num_key_value_groups
                key_transposed = key_states.transpose(2, 3).contiguous()
                attn_weights = torch.matmul(query_states_last, 
                                            repeat_kv(key_transposed, n_rep)) * self.scaling
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, -last_token_num:, :]
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
                    attention_mask = None
                attn_weights = nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)
            else:
                # 进行QK^T计算
                n_rep = self.num_key_value_groups
                attn_weights = torch.matmul(query_states, 
                                            repeat_kv(key_states.transpose(2, 3), n_rep)) * self.scaling

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
                    attention_mask = None
                # upcast attention to fp32
                # 对QK^T进行softmax
                attn_weights = nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)
                # softmax(QK^T)V
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states, n_rep)) 
            # 生成 KV cache, 内容详见 qkv_matmul 函数
            past_key_value = (None,) * 39 + (key_states, value_states, kv_seq_len)
            end_time = time.time()
            # self.decoding_time[self.token_num] = (end_time - start_time)
        
        past_key_value = past_key_value if use_cache else None

        if self.config._attn_implementation != 'flash_attention_2' or any(x is not None for x in past_key_value[0:39]):
            attn_output = attn_output.transpose(1, 2).contiguous()
        # 把32个头的128维重新整合成4096维
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # 存储最近的几个注意力分数
        # 此操作在上下文长度 30k 情况下 prefilling 阶段每层需要 0.2 秒, decoding 阶段每层需要 0.0002 秒
        self.last_weights.update(attn_weights)
        # 如果需要量化
        if self.current_length >= 32:
            start_time = time.time()
            # 压缩的token数是32的倍数
            residual_length = self.current_length % 32
            compress_length = self.current_length - residual_length
            total_length = self.history_reserve_length + compress_length
            block_size = 32
            block_num = total_length // block_size
            token_importance = self.last_weights.calculate_token_importance()
            if auxiliary_parameters is not None:
                compress_ratio = auxiliary_parameters[0]
                K_bit_num = auxiliary_parameters[1]
                V_bit_num = auxiliary_parameters[2]
            else:
                ratio_168 = torch.tensor([64/(self.history_reserve_length+compress_length), 0], dtype=torch.float32)
                compress_ratio = torch.cat((ratio_168, self.ratio_q, self.ratio_e))
                compress_ratio[2:] = (1 - compress_ratio[0]) * compress_ratio[2:]
                K_bit_num = torch.round(block_num * compress_ratio).to(torch.int)
                # 取整之后求和可能和原来不相同，所以应作出一些调整
                K_bit_num = make_sum_equal(K_bit_num, block_num)
                K_bit_num *= block_size
                V_bit_num = K_bit_num.clone()
            eviction_len = K_bit_num[-1]
            sort_importance = token_importance[:self.history_reserve_length + compress_length].clone()
            
            K_token_idx = get_block_token_idx(block_size, block_num, sort_importance)

            past_key_value = KV_cache_hqe_2(past_key_value, compress_length, self.history_reserve_length, K_token_idx, 
                                        K_bit_num, V_bit_num, sort_importance, quant_strategy, self.last_weights, block_size, block_size, self.times_range)
            
            self.history_reserve_length += (compress_length - eviction_len)
            self.current_length -= compress_length

            if auxiliary_parameters is None:
                next_total_lenth = (self.history_reserve_length + 32) // 32 * 32
                add_length = next_total_lenth - self.history_reserve_length
                self.ratio_e = (add_length * self.ratio_e_0) / (self.history_reserve_length + add_length)
                self.ratio_q = (1 - self.ratio_e) * self.ratio_q / self.ratio_q.sum()

            auxiliary_parameters = (compress_ratio, K_bit_num, V_bit_num)
            end_time = time.time()
            if self.token_num < len(self.compress_time):
                self.compress_time[self.token_num] += (end_time - start_time)
        else:
            auxiliary_parameters = None
            
        self.token_num += 1

        attn_weights = attn_weights if output_attentions else None
        return attn_output, attn_weights, past_key_value, auxiliary_parameters
    


class LlamaDecoderLayer_hqe(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention_hqe(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
        auxiliary_parameters: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, auxiliary_parameters = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            auxiliary_parameters=auxiliary_parameters,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, auxiliary_parameters,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_hqe(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layer_num = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_hqe(config, layer_idx) for layer_idx in range(self.layer_num)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.token_num = 0
        self.compress_time = 128 * [0.0]
        self.decoding_time = 128 * [0.0]

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        start_time = time.time()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and isinstance(past_key_values, Cache):
            past_key_values = None

        if cache_position is None:
            past_seen_tokens = past_key_values[0][-1] if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        auxiliary_parameters = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                auxiliary_parameters=auxiliary_parameters,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]
            auxiliary_parameters = layer_outputs[1]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        end_time = time.time()
        if self.token_num < len(self.decoding_time):
            self.decoding_time[self.token_num] = (end_time - start_time)
        self.token_num += 1

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Union[Cache, Tuple[torch.Tensor]]],
        output_attentions: bool,
    ):
        past_seen_tokens = 0
        if past_key_values is not None:
            past_seen_tokens += past_key_values[0][2].shape[2] if past_key_values[0][2] is not None else 0
            past_seen_tokens += past_key_values[0][8].shape[2] if past_key_values[0][8] is not None else 0
            past_seen_tokens += past_key_values[0][16].shape[2] if past_key_values[0][16] is not None else 0
            past_seen_tokens += past_key_values[0][24].shape[2] if past_key_values[0][24] is not None else 0
            past_seen_tokens += past_key_values[0][32].shape[2] if past_key_values[0][32] is not None else 0
            past_seen_tokens += past_key_values[0][40].shape[2] if past_key_values[0][40] is not None else 0

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
       
        target_length = attention_mask.shape[-1] if past_key_values is None else past_seen_tokens + sequence_length

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = target_length
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :mask_length].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    
    def reset_val(self):
        '''把之前的运行中生成的成员变量重置'''
        for _, decoder_layer in enumerate(self.layers):
            decoder_layer.self_attn.compress_length = 0
            decoder_layer.self_attn.current_length = 0
            decoder_layer.self_attn.history_reserve_length = 0
            decoder_layer.self_attn.last_weights.reset()


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class LlamaForCausalLM_hqe(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_hqe(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(self, **kwargs):
        if kwargs.get('max_new_tokens') is not None:
            max_new_tokens = kwargs['max_new_tokens']
            input_length = kwargs['input_ids'].shape[1]
            for _, decoder_layer in enumerate(self.model.layers):
                decoder_layer.self_attn.last_weights.set_max_length(input_length + max_new_tokens)
        elif kwargs.get('max_length') is not None:
            max_length = kwargs['max_length']
            for _, decoder_layer in enumerate(self.model.layers):
                decoder_layer.self_attn.last_weights.set_max_length(max_length)
        output = super().generate(**kwargs)
        self.model.reset_val()
        return output


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
]
