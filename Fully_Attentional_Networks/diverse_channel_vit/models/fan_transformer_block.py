"""
FAN-inspired transformer block with integrated channel attention.
Based on the paper "Understanding The Robustness in Vision Transformers" (ICML 2022).
"""

import torch
import torch.nn as nn
from diverse_channel_vit.models.vit import Block, Attention, Mlp, DropPath
from diverse_channel_vit.models.fan_channel_attention import EnhancedChannelAttention

class FANBlock(nn.Module):
    """
    Transformer block with integrated FAN-inspired channel attention.
    This extends the standard transformer block by adding channel attention
    between the self-attention and feed-forward network.
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        fan_enabled=False,
        fan_num_heads=4,
        fan_dropout=0.0,
        **kwargs
    ):
        super().__init__()
        # Standard transformer components
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # FAN-inspired channel attention
        self.fan_enabled = fan_enabled
        self.channel_diversity_loss = 0.0
        
        if fan_enabled:
            self.norm_fan = norm_layer(dim)
            self.channel_attention = EnhancedChannelAttention(
                embed_dim=dim,
                num_heads=fan_num_heads if fan_num_heads else 4,
                dropout=fan_dropout if fan_dropout else 0.0,
                reduction_ratio=4
            )
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        # Standard self-attention
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
            
        # Apply self-attention with residual connection
        x = x + self.drop_path(y)
        
        # Apply FAN-inspired channel attention if enabled
        if self.fan_enabled:
            # Reshape for channel attention
            # In transformer, x is (B, N, D) where N is sequence length (tokens)
            # We need to reshape to process tokens as "channels" for the FAN attention
            B, N, D = x.shape
            
            # Apply channel attention to normalized features
            x_reshaped = self.norm_fan(x).permute(0, 2, 1)  # (B, D, N)
            
            # Apply channel attention
            scores, diversity_loss = self.channel_attention(x_reshaped)
            
            # Store diversity loss for later use
            self.channel_diversity_loss = diversity_loss
            
            # Apply channel attention scores to features
            # Reshape scores: (B, N) -> (B, N, 1)
            scores = scores.unsqueeze(-1)  # (B, N, 1)
            
            # Apply scores to features (broadcasting across the feature dimension)
            x = x * scores.permute(0, 2, 1)  # Permute to (B, 1, N)
        
        # Apply feed-forward network with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
