import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb


__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']


class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    def _reset_masks(self):
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))


        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)




        # attn_weights (BS, heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
        current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)
        # offset = attn_weights.gt(0).sum(0).sum(1)

        # Accumulate attention scores
        if not self.previous_scores == None:
            current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
        else:
            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.heavy_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])

            # current_scores_sum = current_scores_sum / offset
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum #(heads, k-tokens)
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
    
        if attn_tokens_all > self.cache_budget:
            # activate most recent k-cache
            if not self.recent_budget == 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                # activate historical best self.cache_budget - self.recent_budget tokens.
                # self.previous_scores # (k-Cache - 1)
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if not self.heavy_budget == 0:
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



def convert_kvcache_llama_heavy_recent(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, config)

        if isinstance(module, LlamaAttention):
            model._modules[name] = LlamaAttention_heavy_hitter(config)

    return model

