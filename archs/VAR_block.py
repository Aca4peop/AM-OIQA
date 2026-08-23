

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
    print("Use dropout_add_layer_norm and fused_mlp_func from flash_attn.")
except ImportError: pass

# automatically import faster attention implementations

try: 
    from flash_attn import flash_attn_func         
    print("Use flash_attn_func from flash_attn.")
except ImportError: pass

try: 
    from torch.nn.functional import scaled_dot_product_attention as slow_attn   
    print("Use slow_attn from torch.")
except ImportError: # if there is no flash-attn, use slow_attn
    print("Use slow_attn from torch.")
    
def slow_attn(query, key, value, scale: float, rel_pos_bias=None, attn_mask=None, dropout_p=0.0):
        """
        standard attention implementation
        args:
            query: [B, H, L, C]
            key:   [B, H, S, C]
            value: [B, H, S, C]
            scale: float
            attn_mask: [L, S]
            dropout_p: float
        """
        attn = query.mul(scale) @ key.transpose(-2, -1) 
        if attn_mask is not None: attn.add_(attn_mask)

        if rel_pos_bias is not None: 
            assert rel_pos_bias.shape[-2:] == attn.shape[-2:], "rel_pos_bias shape should be {}, get {}".format(attn.shape, rel_pos_bias.shape)
            attn.add_(rel_pos_bias)
        attn = attn.softmax(dim=-1)
        return (F.dropout(attn, p=dropout_p, inplace=True) if dropout_p > 0 else attn) @ value

class Upsample2x(nn.Module):
    '''
        conv(upsample(x),kernel_size=3,stride=1,padding=1)
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class Downsample2x(nn.Module):
    """
        conv(pad(x,(0,1,0,1)),kernel_size=3,stride=2,padding=0)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))

class FFN(nn.Module):
    """
    infeatures -> hidden_features -> out_features
    drop_rate : dropout rate of the dropout layer
    fused_if_available: if fused_mlp_func is available, otherwise use nn.Linear
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate, inplace=True) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'

class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        """
        the self attention block used in VAR transformer, with support for flash attention and L2 normalization.
            block_idx (int): the index of the attention block.
            embed_dim (int, optional): the embedding dimension of the input tensor.
            num_heads (int, optional): the number of attention heads, default is 12.
            attn_drop (float, optional): the dropout rate for attention weights, default is 0.
            proj_drop (float, optional): the dropout rate for the projection layer, default is 0.
            attn_l2_norm (bool, optional): whether to use L2 normalization, default is False.
            flash_if_available (bool, optional): whether to use Flash Attention if available, default is True.
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        # the index of the attention block, the number of attention heads, the dimension of each attention head
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  
        # if L2 normalization is enabled, then scale is 1 and scale_mul_1H11 is a learnable parameter
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log())
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        
        # if kv caching is enabled, then cached_k and cached_v will be used instead of qkv
        self.caching, self.cached_k, self.cached_v = False, None, None

        self.extra_repr()
    # set kv caching on/off
    def kv_caching(self, enable: bool): 
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias=None):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = torch.float16
        
        using_flash = self.using_flash and attn_bias is None
        if using_flash : q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q k v:BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q k v: BHLc
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        oup = oup.to(dtype = x.dtype)
        return self.proj_drop(self.proj(oup))
        
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, attn_l2_norm={self.attn_l2_norm}'

class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, embed_dim, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=True, fused_if_available=True,
    ):
       
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx= block_idx
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, 
                                  num_heads=num_heads, attn_drop=attn_drop, 
                                  proj_drop=drop, attn_l2_norm=attn_l2_norm, 
                                  flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio),drop_rate=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)        
        self.fused_add_norm_fn = None
    
    def forward(self, x, attn_bias=None): 
        x = x + self.drop_path(self.attn(self.ln_wo_grad(x), attn_bias=attn_bias ))
        x = x + self.drop_path(self.ffn(self.ln_wo_grad(x))) 
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    # calculate the probability of keeping the element, and scale the input accordingly
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    # sample with bernoulli distribution, where each element is 1 with probability keep_prob
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'