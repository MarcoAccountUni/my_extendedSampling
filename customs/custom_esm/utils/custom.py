import math
import torch


# Efficient implementation of mean_dot_product_attention, which is a variant of 
# torch.nn.functional.scaled_dot_product_attention() to compute the mean attention matrix out of all the heads.
# To see the original function: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def mean_dot_product_attention(
        query, key, value, attn_mask=None, 
        only_attention=False,
        dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)            
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale            
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)            
    if is_causal:            
        assert attn_mask is None            
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)            
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))            
        attn_bias.to(query.dtype)            

    if attn_mask is not None:            
        if attn_mask.dtype == torch.bool:            
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))            
        else:            
            attn_bias = attn_mask + attn_bias            

    if enable_gqa:            
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)            
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)            

    attn_weight = query @ key.transpose(-2, -1) * scale_factor            
    attn_weight += attn_bias            
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight.mean(axis=1) # B x H x L x L -> B x L x L


# Clean (remove EOS and BOS columns and rows) and symmetrize the attention matrix
def clean_and_symm(attn):
    attn = attn[:, 1:-1, 1:-1]
    attn = (attn + attn.transpose(-1,-2)) / 2.
    return attn
