import re

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils.deprecation_utils import deprecate

from typing import Optional
from einops import rearrange, repeat

from masactrl.utils import attn_adain


def set_masactrl_attn(model):
    attn_processor = model.attn_processors
    pattern = "(up|mid)\w.*attn1"

    masactrl_processor = MasaProcessor()
    for k, v in attn_processor.items():
        if re.match(pattern, k) is not None:
            attn_processor[k] = masactrl_processor

    model.set_attn_processor(attn_processor)


class MasaProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        # attn_dims = [16]
        # self.attn_masks={}
        # for attn_dim in attn_dims:
        #     attn_mask = torch.zeros(6*(attn_dim)**2, 6*(attn_dim)**2)
        #     for i in range(3):
        #         for j in range(2):
        #             temp = torch.zeros(attn_dim*3, attn_dim*2)
        #             temp[i*attn_dim: (i+1)*attn_dim, j*attn_dim:(j+1)*attn_dim] = 1
        #             temp = torch.flatten(temp).unsqueeze(0).T * torch.flatten(temp).unsqueeze(0)
        #             attn_mask += temp

        #     attn_mask = torch.stack([torch.ones_like(attn_mask), attn_mask, torch.ones_like(attn_mask), attn_mask])
        #     attn_mask = attn_mask.type(torch.bool).unsqueeze(1).cuda().requires_grad_(False)
        #     self.attn_masks[attn_dim] = attn_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # masactrl
        query[-1:] = query[-2:-1]
        # key = key[:1].repeat(2,1,1,1)
        key = torch.cat([key[:1], key[1:2], key[:1], key[:1]])
        # value = value[:1].repeat(2,1,1,1)
        value = torch.cat([value[:1], value[1:2], value[:1], value[1:2]])

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
