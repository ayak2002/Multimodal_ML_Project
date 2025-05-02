"""
FAN-inspired channel attention mechanisms for improving robustness to unseen channels.
Based on the paper "Understanding The Robustness in Vision Transformers" (ICML 2022).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class CrossChannelAttention(nn.Module):
    """
    Cross-channel attention module inspired by FAN (Fully Attentional Networks).
    This module allows information sharing across channels to build more robust representations.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention for cross-channel communication
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Channel embeddings of shape (B, C, D) where:
               B is batch size, C is number of channels, D is embedding dimension
        Returns:
            Enhanced channel embeddings with cross-channel information
        """
        B, C, D = x.shape
        
        # Reshape for multi-head attention
        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, C, head_dim)
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, C, C)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, C, D)  # (B, C, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class EnhancedChannelAttention(nn.Module):
    """
    Enhanced channel attention module that combines the original channel attention
    with cross-channel information sharing for improved robustness.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.0, reduction_ratio=4):
        super().__init__()
        
        # Cross-channel attention for information sharing
        self.cross_channel_attn = CrossChannelAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Channel scoring mechanism (similar to original implementation but with enhanced features)
        self.channel_scoring = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // reduction_ratio),
            nn.GELU(),  # Using GELU instead of ReLU for better gradient flow
            nn.Linear(embed_dim // reduction_ratio, 1),
            nn.Sigmoid()
        )
        
        # Channel diversity regularization
        self.diversity_weight = 0.1
        
    def forward(self, x):
        """
        Args:
            x: Channel embeddings of shape (B, C, D) where:
               B is batch size, C is number of channels, D is embedding dimension
        Returns:
            Channel attention scores and diversity loss
        """
        # Apply cross-channel attention to enhance representations
        x_enhanced = self.cross_channel_attn(x)
        
        # Add residual connection
        x = x + x_enhanced
        
        # Generate channel attention scores
        scores = self.channel_scoring(x).squeeze(-1)  # (B, C)
        
        # Calculate channel diversity loss
        diversity_loss = self._calculate_diversity_loss(x)
        
        return scores, diversity_loss * self.diversity_weight
    
    def _calculate_diversity_loss(self, x):
        """Calculate diversity loss to encourage diverse channel representations"""
        # Normalize features
        x_norm = F.normalize(x, p=2, dim=-1)
        
        # Calculate channel similarity matrix
        channel_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, C, C)
        
        # Mask out diagonal elements
        mask = torch.eye(channel_sim.size(1), device=channel_sim.device).unsqueeze(0)
        channel_sim = channel_sim * (1 - mask)
        
        # Calculate diversity loss (lower similarity = higher diversity)
        diversity_loss = torch.mean(channel_sim)
        
        return diversity_loss


class ChannelAdaptationModule(nn.Module):
    """
    Module for adapting to novel channels at test time.
    Uses a bank of known channel embeddings to adapt to unseen channels.
    """
    def __init__(self, num_channels, embed_dim):
        super().__init__()
        self.adaptation_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        
    def forward(self, channel_embs, available_channels=None):
        """
        Args:
            channel_embs: Channel embeddings of shape (C, D)
            available_channels: List of available channel indices
        Returns:
            Adapted channel embeddings
        """
        # Generate adaptation weights
        adaptation_weights = self.adaptation_network(channel_embs)
        
        # Add residual connection
        adapted_embs = channel_embs + adaptation_weights
        
        return adapted_embs
