import numpy as np
import torch
from torch import nn
from torch import tensor
import sys
from einops import rearrange
from config import Model
from collections import Counter
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional
from helper_classes.first_layer_init import NewChannelLeaveOneOut, FirstLayerInit
from functools import partial
from utils import trunc_normal_
from models.vit import Block, BlockV2
import random
# ... Other imports ...
from utils import trunc_normal_
from models.vit import Block, BlockV2
from collections import defaultdict
import torch
from einops import repeat
from torch import tensor
import torch.distributed as dist
import torch.nn as nn
import os
import sys
from models.loss_fn import *
import math
import random
from functools import partial
from typing import List, Dict, Optional
from helper_classes.first_layer_init import NewChannelLeaveOneOut
import numpy as np
import torch
from torch import nn
import sys
from einops import rearrange
from config import Model
from collections import Counter
import torch.nn.functional as F
from einops import rearrange, repeat

class ChannelStatisticsEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.stats_encoder = nn.Sequential(
            nn.Linear(5, 32),  # 5 statistical features per channel
            nn.GELU(),
            nn.Linear(32, embed_dim),
            nn.Dropout(dropout)
        )
        
    def compute_channel_statistics(self, x):
        """Compute statistical properties of a channel"""
        # x shape: [B, H, W]
        batch_size = x.shape[0]
        
        # Calculate statistics (shape: [B, 5])
        mean = x.mean(dim=[1, 2], keepdim=False)
        std = x.std(dim=[1, 2], keepdim=False)
        min_val = x.view(batch_size, -1).min(dim=1)[0]
        max_val = x.view(batch_size, -1).max(dim=1)[0]
        median = x.view(batch_size, -1).median(dim=1)[0]
        
        # Combine statistics
        statistics = torch.stack([mean, std, min_val, max_val, median], dim=1)
        return statistics
    
    def forward(self, x):
        """
        Forward pass for statistics encoder
        Args:
            x: Input tensor of shape [B, H, W]
        Returns:
            Statistics embedding of shape [B, embed_dim]
        """
        stats = self.compute_channel_statistics(x)
        return self.stats_encoder(stats)


class ChannelAdapter(nn.Module):
    """
    Meta-network for adapting to novel channels
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass for channel adapter
        Args:
            x: Input tensor of shape [B, H, W]
        Returns:
            Adapter embedding of shape [B, embed_dim]
        """
        x = x.unsqueeze(1)  # Add channel dimension [B, 1, H, W]
        return self.adapter(x)


class ChannelAttention(nn.Module):
    """Cross-channel attention mechanism to allow information sharing between channels"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [B, C, dim]
        batch_size, num_channels, _ = x.shape
        
        # Project to q, k, v
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b c (h d) -> b h c d', h=self.num_heads)
        k = rearrange(k, 'b c (h d) -> b h c d', h=self.num_heads)
        v = rearrange(v, 'b c (h d) -> b h c d', h=self.num_heads)
        
        # Compute attention scores
        scale = (self.dim // self.num_heads) ** -0.5
        attention = torch.matmul(q, k.transpose(-1, -2)) * scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h c d -> b c (h d)')
        
        # Project to output
        return self.to_out(out)


class RobustPatchEmbedPerChannel(nn.Module):
    """Enhanced PatchEmbedPerChannel with robust novel channel handling."""

    def __init__(
        self,
        config,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        mapper: Dict | None = None,
        embed_dim: int = 768,
        enable_sample: bool = True,
        use_channelvit_channels: bool = True,
    ):
        super().__init__()
        self.cfg = config
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.mapper = mapper
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.channel_scale = np.sqrt(1.0 / self.cfg.temperature)
        
        # Channel embedding components from original implementation
        if self.cfg.proxy_loss_lambda > 0:
            self.channel_emb_proxies = torch.nn.Parameter((torch.randn(in_chans, embed_dim) / 8))
            if self.cfg.get("proxy_orthogonal_init", False):
                nn.init.orthogonal_(self.channel_emb_proxies)
        if self.cfg.hcs_sampling != "none" and self.cfg.hcs_sampling is not None:
            self.counter = defaultdict(lambda: 0)

        if self.cfg.hcs_sampling.endswith("resnet34"):
            import timm
            self.resnet34 = timm.create_model("resnet34", pretrained=True, num_classes=0)
            ## turn off grad
            for param in self.resnet34.parameters():
                param.requires_grad = False
            self.resnet34.eval()

        # Patch embedding projection
        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        
        # Channel embeddings
        if use_channelvit_channels:
            self.channel_embed = nn.Embedding(in_chans, embed_dim)
            if self.cfg.orthogonal_channel_emb_init:
                print("---------------- use_channelvit_channels=True")
                nn.init.orthogonal_(self.channel_embed.weight)
            else:
                trunc_normal_(self.channel_embed.weight, std=0.02)

            if self.cfg.freeze_channel_emb:
                self.channel_embed.weight.requires_grad = False
                print("---------- Froze channel embedding!")
        
        # NEW: Components for handling novel channels
        self.stats_encoder = ChannelStatisticsEncoder(embed_dim)
        self.channel_adapter = ChannelAdapter(embed_dim)
        
        # NEW: Fusion module for combining static and dynamic embeddings
        self.channel_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Channel similarity bank for finding closest matches
        self.register_buffer('channel_similarity_bank', None, persistent=False)
        self.register_buffer('channel_features_bank', None, persistent=False)

        self.use_channelvit_channels = use_channelvit_channels
        self.enable_sample = enable_sample
        
        # NEW: Flag to control which novel channel strategy to use
        self.novel_channel_strategy = getattr(config, 'novel_channel_strategy', 'dynamic_features')
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.6)
        self.emb_interp_factor = getattr(config, 'emb_interp_factor', 0.7)

    def get_channel_emb_resnet34(self, x):
        ##   use resnet34
        num_channels = x.shape[1]
        out = []
        for i in range(num_channels):
            x_i = x[:, i, :, :]
            x_i = repeat(x_i, "b h w -> b c h w", c=3)
            output_i = self.resnet34(x_i)
            out.append(output_i)
        out = torch.stack(out, dim=1)
        return out  # (b, num_channels, z_dim)

    def compute_channel_similarity(self, channel_data, reference_data):
        """
        Compute similarity between a channel and reference channels with numerical stability
        """
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Flatten spatial dimensions
        channel_flat = channel_data.flatten(1)  # [B, H*W]
        reference_flat = reference_data.flatten(1)  # [N, H*W]
        
        # Normalize with epsilon for stability
        channel_norm = torch.norm(channel_flat, p=2, dim=1, keepdim=True).clamp(min=eps)
        reference_norm = torch.norm(reference_flat, p=2, dim=1, keepdim=True).clamp(min=eps)
        
        channel_flat = channel_flat / channel_norm
        reference_flat = reference_flat / reference_norm
        
        # Compute cosine similarity
        similarity = torch.matmul(channel_flat, reference_flat.t())  # [B, N]
        
        # Clip to valid range and average across batch
        similarity = torch.clamp(similarity, -1.0, 1.0)
        return similarity.mean(dim=0)  # Average across batch [N]

    def get_novel_channel_embedding(self, channel_data, chunk_name, training_channels, device):
        """
        Generate embedding for novel channels using various strategies
        Args:
            channel_data: [B, H, W] - Channel data
            chunk_name: Current chunk name
            training_channels: List of channels seen during training
            device: Computation device
        Returns:
            Channel embedding
        """
        batch_size = channel_data.shape[0]
        
        if self.novel_channel_strategy == 'dynamic_features':
            # Use both statistical features and visual features
            stats_emb = self.stats_encoder(channel_data)
            visual_emb = self.channel_adapter(channel_data)
            
            # Combine both embeddings
            combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
            channel_emb = self.channel_fusion(combined_emb)
            
        elif self.novel_channel_strategy == 'nearest_neighbor':
            # Find most similar training channel and use its embedding
            if self.channel_features_bank is not None and len(training_channels) > 0:
                # Compute similarity to all channels in the bank
                similarities = self.compute_channel_similarity(
                    channel_data, self.channel_features_bank)
                
                # Get the most similar channel
                most_similar_idx = similarities.argmax().item()
                if most_similar_idx < len(training_channels):  # Safety check
                    similar_channel = training_channels[most_similar_idx]
                    
                    # Use embedding of most similar channel
                    channel_emb = self.channel_embed(tensor([similar_channel], device=device)).repeat(batch_size, 1)
                else:
                    # Fallback to dynamic features
                    stats_emb = self.stats_encoder(channel_data)
                    visual_emb = self.channel_adapter(channel_data)
                    combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                    channel_emb = self.channel_fusion(combined_emb)
            else:
                # Fallback to dynamic features if no bank is available
                stats_emb = self.stats_encoder(channel_data)
                visual_emb = self.channel_adapter(channel_data)
                combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                channel_emb = self.channel_fusion(combined_emb)
                
        elif self.novel_channel_strategy == 'weighted_interpolation':
            # Use weighted average of similar channel embeddings
            if self.channel_features_bank is not None and self.channel_similarity_bank is not None and len(training_channels) > 0:
                # Compute similarity to all channels in the bank
                similarities = self.compute_channel_similarity(
                    channel_data, self.channel_features_bank)
                
                # Get top-k similar channels (safely)
                k = min(3, max(1, len(training_channels)))
                if k > 0:
                    top_k_vals, top_k_indices = torch.topk(similarities, k)
                    
                    # Only use channels with similarity above threshold
                    valid_mask = top_k_vals > self.similarity_threshold
                    if valid_mask.any():
                        # Normalize weights for valid channels with softmax and clamp
                        weights = F.softmax(top_k_vals[valid_mask].clamp(min=-10, max=10), dim=0)
                        valid_indices = top_k_indices[valid_mask].tolist()
                        
                        # Get embeddings for top channels with bounds checking
                        valid_channel_ids = []
                        for idx in valid_indices:
                            if idx < len(training_channels):
                                valid_channel_ids.append(training_channels[idx])
                        
                        if len(valid_channel_ids) > 0:
                            channel_ids_tensor = tensor(valid_channel_ids, device=device)
                            channel_embs = self.channel_embed(channel_ids_tensor)
                            
                            # Weighted average of embeddings (safe against NaN)
                            weighted_emb = torch.matmul(weights[:len(valid_channel_ids)], channel_embs)
                            
                            # Combine with dynamic features
                            stats_emb = self.stats_encoder(channel_data)
                            visual_emb = self.channel_adapter(channel_data)
                            dynamic_emb = self.channel_fusion(torch.cat([stats_emb, visual_emb], dim=1))
                            
                            # Interpolate between weighted static and dynamic embeddings
                            factor = torch.clamp(torch.tensor(self.emb_interp_factor), 0.0, 1.0)
                            channel_emb = (
                                factor * weighted_emb + 
                                (1 - factor) * dynamic_emb
                            )
                            channel_emb = channel_emb.repeat(batch_size, 1)
                        else:
                            # Fallback to dynamic features
                            stats_emb = self.stats_encoder(channel_data)
                            visual_emb = self.channel_adapter(channel_data)
                            combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                            channel_emb = self.channel_fusion(combined_emb)
                    else:
                        # Fallback to dynamic features
                        stats_emb = self.stats_encoder(channel_data)
                        visual_emb = self.channel_adapter(channel_data)
                        combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                        channel_emb = self.channel_fusion(combined_emb)
                else:
                    # Fallback to dynamic features
                    stats_emb = self.stats_encoder(channel_data)
                    visual_emb = self.channel_adapter(channel_data)
                    combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                    channel_emb = self.channel_fusion(combined_emb)
            else:
                # Fallback to dynamic features if no bank is available
                stats_emb = self.stats_encoder(channel_data)
                visual_emb = self.channel_adapter(channel_data)
                combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                channel_emb = self.channel_fusion(combined_emb)
        else:
            # Default to original implementation's approach with safety
            if len(training_channels) >= 3:
                c1 = random.choice(training_channels)
                c2 = random.choice(training_channels)
                c3 = random.choice(training_channels)
                idx = tensor([c1, c2, c3], device=device)
                channel_emb = self.channel_embed(idx).mean(dim=0, keepdim=True).repeat(batch_size, 1)
            elif len(training_channels) > 0:
                # Sample with replacement if fewer than 3 channels
                idx = tensor([random.choice(training_channels) for _ in range(3)], device=device)
                channel_emb = self.channel_embed(idx).mean(dim=0, keepdim=True).repeat(batch_size, 1)
            else:
                # Fallback to dynamic features
                stats_emb = self.stats_encoder(channel_data)
                visual_emb = self.channel_adapter(channel_data)
                combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                channel_emb = self.channel_fusion(combined_emb)
            
        # Final check to avoid NaN values
        if torch.isnan(channel_emb).any():
            # Create a fallback embedding
            emb_dim = channel_emb.shape[-1]
            channel_emb = torch.zeros(batch_size, emb_dim, device=device)
            
        return channel_emb
    
    def update_channel_banks(self, x, chunk_name):
        """
        Update the channel feature banks for similarity comparison
        Args:
            x: Input tensor of shape [B, C, H, W]
            chunk_name: Current chunk name
        """
        cur_channels = self.mapper[chunk_name]
        features_list = []
        
        # Extract features for each channel
        for i, channel_id in enumerate(cur_channels):
            channel_data = x[:, i]  # [B, H, W]
            features_list.append(channel_data.detach().mean(dim=0, keepdim=True))  # [1, H, W]
        
        # Stack features for all channels
        channel_features = torch.cat(features_list, dim=0)  # [C, H, W]
        
        # Update banks
        if self.channel_features_bank is None:
            self.channel_features_bank = channel_features
            self.channel_similarity_bank = torch.arange(len(cur_channels))
        else:
            # Concatenate with existing bank
            self.channel_features_bank = torch.cat([self.channel_features_bank, channel_features], dim=0)
            self.channel_similarity_bank = torch.cat([
                self.channel_similarity_bank, 
                torch.arange(len(cur_channels)) + len(self.channel_similarity_bank)
            ])

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks,
        new_channel_init: NewChannelLeaveOneOut | None,
        extra_tokens={},
        **kwargs,
    ):
        # assume all images in the same batch has the same input channels
        cur_channels = self.mapper[chunk_name]  ## type: ignore
        if self.use_channelvit_channels:
            channel_embed = self.channel_embed(tensor(cur_channels, device=x.device))  #  Cin, embed_dim=Cout
        b, Cin, h, w = x.shape
        Cin_original = Cin
        
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans
        ## if training time, and we use channel sampling
        if self.training and self.enable_sample:
            Cin_new = random.randint(1, Cin)

            if self.cfg.hcs_sampling == "none" or self.cfg.hcs_sampling is None:
                cur_channels = random.sample(cur_channels, k=Cin_new)
                Cin = Cin_new
                channels_idx = [self.mapper[chunk_name].index(c) for c in cur_channels]
                x = x[:, channels_idx, :, :]
                if self.use_channelvit_channels:
                    channel_embed = channel_embed[channels_idx]
            elif self.cfg.hcs_sampling == "hcs_per_sample":
                channels_idxs = []
                for _ in range(b):
                    tmp = random.sample(cur_channels, k=Cin_new)
                    channels_idx = [self.mapper[chunk_name].index(c) for c in tmp]
                    channels_idxs.append(channels_idx)
                Cin = Cin_new
                channels_idxs_tensor = torch.tensor(channels_idxs, device=x.device)
                channel_embed_expand = repeat(channel_embed, "Cin Cout -> B Cin Cout", B=b)
                first_idxs = torch.arange(b)[:, None]
                channel_embed_kept = channel_embed_expand[first_idxs, channels_idxs_tensor]
                x = x[first_idxs, channels_idxs_tensor]
            else:
                # Code kept from original implementation
                assert (
                    self.use_channelvit_channels
                ), "hcs_sampling only works with use_channelvit_channels=True"
                with torch.no_grad():
                    first_channel_idx = random.randint(0, Cin - 1)

                    if self.cfg.hcs_sampling.endswith("_proj"):
                        x_sim = self.proj(x.unsqueeze(1))
                        x_sim = rearrange(x_sim, "b d c h w -> b c (h w d)")
                        x_sim = F.normalize(x_sim, p=2, dim=-1)  ## b, c, d
                        cosine_sim = torch.einsum("b c d, b e d -> b c e", x_sim, x_sim).mean(dim=0)
                        cosine_scores = cosine_sim[first_channel_idx]
                    elif self.cfg.hcs_sampling == "lowest_cosine_prob_resnet34":
                        out = self.get_channel_emb_resnet34(x)  ## b, num_channel, resnet_dim
                        out = F.normalize(out, p=2, dim=-1)
                        cosine_sim = torch.einsum("b c d, b e d -> b c e", out, out).mean(dim=0)
                        cosine_scores = cosine_sim[first_channel_idx]
                    else:
                        # channel_embed_cor = torch.einsum("c d, e d -> c e", channel_embed, channel_embed)
                        channel_emb_norm = F.normalize(channel_embed, p=2, dim=-1)
                        channel_embed_cosine = torch.einsum(
                            "c d, e d -> c e", channel_emb_norm, channel_emb_norm
                        )
                        ## get the cosine similarity between the first channel and the rest
                        cosine_scores = channel_embed_cosine[first_channel_idx]

                    if self.cfg.hcs_sampling == "lowest_cosine":
                        _, indices = torch.topk(cosine_scores, k=Cin_new, largest=False)
                        indices = indices.cpu().numpy().tolist()
                        if first_channel_idx not in indices:
                            indices[-1] = first_channel_idx
                        cur_channels = [cur_channels[i] for i in indices]
                    elif self.cfg.hcs_sampling == "highest_cosine":
                        _, indices = torch.topk(cosine_scores, k=Cin_new, largest=True)
                        indices = indices.cpu().numpy().tolist()
                        if first_channel_idx not in indices:
                            indices[-1] = first_channel_idx
                        cur_channels = [cur_channels[i] for i in indices]
                    elif self.cfg.hcs_sampling in [
                        "lowest_cosine_prob",
                        "lowest_cosine_prob_proj",
                        "lowest_cosine_prob_resnet34",
                    ]:

                        scores = (1 - cosine_scores) / self.cfg.hcs_sampling_temp
                        ## make the dist more peaky
                        prob = F.softmax(scores, dim=-1)

                        ## sample Cin_new channels without replacement
                        indices = torch.multinomial(prob, Cin_new, replacement=False)
                        indices = indices.cpu().numpy().tolist()
                        if first_channel_idx not in indices:
                            indices[-1] = first_channel_idx
                        cur_channels = [cur_channels[i] for i in indices]
                    else:
                        ### cosine is only use of prob, not absolute
                        raise ValueError(f"Invalid hcs_sampling: '{self.cfg.hcs_sampling}'")

                Cin = Cin_new
                channels_idx = [self.mapper[chunk_name].index(c) for c in cur_channels]
                x = x[:, channels_idx, :, :]
                if self.use_channelvit_channels:
                    channel_embed = channel_embed[channels_idx]

                counter = Counter(cur_channels)
                for k, v in counter.items():
                    self.counter[k] += v
        
        # MODIFIED: Handling novel channels at test time
        if self.use_channelvit_channels and (not self.training):
            if self.training and getattr(self.cfg, 'update_channel_bank', False):
                # Update channel feature bank during training
                self.update_channel_banks(x, chunk_name)
                
            if training_chunks is not None:
                training_chunks = training_chunks.split("_")
                training_channels = [self.mapper[ch] for ch in training_chunks]
                training_channels = [item for sublist in training_channels for item in sublist]  ## flatten
                
                # Process each channel
                param_list = []
                
                for i, c in enumerate(self.mapper[chunk_name]):
                    if c not in training_channels:
                        # Novel channel detected - use our robust strategy
                        channel_data = x[:, i]  # [B, H, W]
                        
                        # Get embedding for novel channel
                        channel_emb = self.get_novel_channel_embedding(
                            channel_data, chunk_name, training_channels, x.device)
                        
                        param_list.append(channel_emb)
                    else:
                        # Known channel - use the standard embedding
                        idx = tensor([c], device=x.device)
                        param = self.channel_embed(idx)
                        param = repeat(param, "1 emb -> b emb", b=b)
                        param_list.append(param)
                
                # Stack all channel embeddings
                channel_embed = torch.stack(param_list, dim=1)  # [B, C, emb_dim]

        # rest of the implementation follows the original code
        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W
        if self.cfg.ortho_loss_v1_lambda > 0:
            n_patches = x.shape[3] * x.shape[4]
            token_labels = torch.arange(x.shape[2]).repeat_interleave(n_patches).to(x.device)
            x_reshaped = rearrange(x, "B Cout Cin H W -> B (Cin H W) Cout").clone()
            orthoproj_loss = ortho_proj_loss_fn_v2(
                x_reshaped,
                labels=token_labels,
                gamma_s=self.cfg.gamma_s,
                gamma_d=self.cfg.gamma_d,
                reverse_pos_pairs=self.cfg.reverse_pos_pairs,
                use_square=self.cfg.use_square,
            )
        else:
            orthoproj_loss = 0

        # channel specific offsets
        if self.cfg.hcs_sampling == "hcs_per_sample":
            raise ValueError("hcs_per_sample not implemented!")

        ## create one hot ground truth for the channel
        ## make ground true for CE loss, with shape B, Cin
        if self.cfg.proxy_loss_lambda > 0:
            channel_gt = torch.eye(Cin, device=x.device)
            channel_emb_proxies = self.channel_emb_proxies[cur_channels]
            proxyloss = proxy_loss(channel_emb_proxies, channel_embed, channel_gt, scale=self.channel_scale)
        else:
            proxyloss = 0

        ortho_proxy_loss = (
            orthoproj_loss * self.cfg.ortho_loss_v1_lambda + proxyloss * self.cfg.proxy_loss_lambda
        )
        
        if self.use_channelvit_channels:
            if len(channel_embed.shape) == 2:  # [Cin, Cout]
                channel_embed = repeat(channel_embed, "Cin Cout -> B Cout Cin", B=x.shape[0])
            # If channel_embed is already [B, C, emb_dim], we don't need to reshape
            x += channel_embed.unsqueeze(-1).unsqueeze(-1)

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout

        return x, Cin, ortho_proxy_loss


class RobustChannelVisionTransformer(nn.Module):
    """Channel Vision Transformer with enhanced novel channel handling"""

    def __init__(
        self,
        config,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        mapper: Dict | None = None,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_sample=False,
        use_channelvit_channels=True,
        **kwargs,
    ):
        super().__init__()
        self.cfg = config
        drop_path_rate = config.drop_path_rate
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans

        # Replace original PatchEmbedPerChannel with our robust version
        self.patch_embed = RobustPatchEmbedPerChannel(
            config=config,
            img_size=img_size[0],
            patch_size=patch_size,
            mapper=mapper,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample,
            use_channelvit_channels=use_channelvit_channels,
        )
        
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        print("----dpr", dpr)

        if self.cfg.block_type == "block_v2":
            BlockClass = BlockV2
        elif self.cfg.block_type == "block":
            BlockClass = Block
        else:
            raise ValueError(f"Unknown block type: {self.cfg.block_type}")
        self.blocks = nn.ModuleList(
            [
                BlockClass(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, nc):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, nc, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, chunk: str, training_chunks_str, new_channel_init, extra_tokens):
        B, _, w, h = x.shape
        x, nc, ortho_proxy_loss = self.patch_embed(
            x, chunk, training_chunks_str, new_channel_init, extra_tokens
        )  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h, nc)

        ### drop some tokens randomly at the last dim
        if self.cfg.dropout_tokens_hcs == "random" and self.training:  ## x: [B CinHW Cout]
            cinHW = x.shape[1]
            HW = cinHW // nc
            cinHW_new = random.randint(1, nc) * HW
            drops_true = random.sample(range(cinHW), k=cinHW_new)
            ## make sure the first token ([CLS]) is not dropped
            drops = [True]
            for i in range(1, cinHW):
                if i in drops_true:
                    drops.append(True)
                else:
                    drops.append(False)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]
        elif self.cfg.dropout_tokens_hcs == "channel" and self.training:
            cinHW = x.shape[1]
            HW = cinHW // nc
            cin_new = random.randint(1, nc)
            ## choose cin_new from nc channels
            drops_channels = random.sample(range(nc), k=cin_new)
            drops = [True]  ## make sure the first token ([CLS]) is not dropped
            for i in range(nc):
                if i in drops_channels:
                    tmp = [True] * HW
                else:
                    tmp = [False] * HW
                drops.extend(tmp)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]
        elif self.cfg.dropout_tokens_hcs == "channel_random50" and self.training:
            cinHW = x.shape[1]
            HW = cinHW // nc
            ## get ceil(50% of the channels)
            cin_new = int(math.ceil(0.5 * nc))
            ## choose cin_new from nc channels
            drops_channels = random.sample(range(nc), k=cin_new)
            drops = [True]  ## make sure the first token ([CLS]) is not dropped
            for i in range(nc):
                if i in drops_channels:
                    tmp = [True] * HW
                else:
                    tmp = [False] * HW
                drops.extend(tmp)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]
        elif self.cfg.dropout_tokens_hcs == "token_random50" and self.training:  ## x: [B CinHW Cout]
            cinHW = x.shape[1]
            HW = cinHW // nc
            cinHW_new = int(math.ceil(0.5 * nc)) * HW
            drops_true = random.sample(range(cinHW), k=cinHW_new)
            ## make sure the first token ([CLS]) is not dropped
            drops = [True]
            for i in range(1, cinHW):
                if i in drops_true:
                    drops.append(True)
                else:
                    drops.append(False)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]
            
        # NEW: Add channel diversity regularization if enabled
        # NEW: Add channel diversity regularization if enabled
        if getattr(self.cfg, 'channel_diversity_reg', False) and self.training:
            # Extract embeddings for different channels
            # Skip the CLS token
            token_embeddings = x[:, 1:, :]
            
            # Reshape to get per-channel embeddings
            hw = (w // self.patch_size) * (h // self.patch_size)
            if nc > 0 and hw > 0:  # Check to avoid division issues
                channel_embeddings = token_embeddings.view(B, nc, hw, -1).mean(dim=2)  # [B, nc, dim]
                
                # Normalize embeddings with epsilon
                norm = torch.norm(channel_embeddings, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                channel_embeddings = channel_embeddings / norm
                
                # Compute pairwise cosine similarity
                similarity = torch.bmm(channel_embeddings, channel_embeddings.transpose(1, 2))  # [B, nc, nc]
                similarity = torch.clamp(similarity, -1.0, 1.0)  # Ensure valid range
                
                # Create identity matrix to mask out self-similarity
                identity = torch.eye(nc, device=x.device).unsqueeze(0).expand(B, -1, -1)
                
                # Get mean similarity of each channel to others (excluding self)
                masked_similarity = similarity * (1 - identity)
                
                # Safe division with non-zero denominator
                denominator = B * nc * (nc - 1)
                if denominator > 0:
                    diversity_loss = masked_similarity.sum() / denominator
                    
                    # Add to existing loss with a weight factor
                    diversity_weight = getattr(self.cfg, 'diversity_weight', 0.1)
                    ortho_proxy_loss = ortho_proxy_loss + diversity_weight * diversity_loss

        return self.pos_drop(x), ortho_proxy_loss

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks: str | None = None,
        new_channel_init: NewChannelLeaveOneOut | None = None,
        extra_tokens={},
    ):
        B, _, w, h = x.shape
        x, ortho_proxy_loss = self.prepare_tokens(
            x, chunk_name, training_chunks, new_channel_init, extra_tokens
        )
        nc = x.shape[1] // ((w // self.patch_size) * (h // self.patch_size))

        # Store intermediate attention maps if requested
        attention_maps = []
        store_attention = getattr(self.cfg, 'store_attention_maps', False)
        
        for i, blk in enumerate(self.blocks):
            if isinstance(blk, BlockV2):
                if store_attention:
                    x, counter, attn = blk(x, pruning_method=self.cfg.dropout_tokens_hcs, nc=nc, return_attention=True)
                    attention_maps.append(attn)
                else:
                    x, counter = blk(x, pruning_method=self.cfg.dropout_tokens_hcs, nc=nc)
            else:
                if store_attention:
                    x, attn = blk(x, return_attention=True)
                    attention_maps.append(attn)
                else:
                    x = blk(x)

        x = self.norm(x)
        
        # Return with attention maps if requested
        if store_attention:
            return x[:, 0].clone(), ortho_proxy_loss, attention_maps
        else:
            return x[:, 0].clone(), ortho_proxy_loss

    def get_last_selfattention(self, x, extra_tokens={}, chunk="", layer_idx=-1):
        x, _ = self.prepare_tokens(
            x, chunk=chunk, training_chunks_str=None, new_channel_init=None, extra_tokens=extra_tokens
        )

        for i, blk in enumerate(self.blocks):
            if i == layer_idx:
                return blk(x, return_attention=True)

            x = blk(x)

    def get_intermediate_layers(self, x, extra_tokens={}, n=1):
        x = self.prepare_tokens(x, extra_tokens)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

def channelvit_distill(config, patch_size=14, in_chans=0, mapper=None, **kwargs):
    model = RobustChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        mapper=mapper,
        in_chans=in_chans,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_tiny(config, patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = RobustChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        in_chans=in_chans,
        mapper=mapper,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_small(config, patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = RobustChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        in_chans=in_chans,
        mapper=mapper,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_base(config, patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = RobustChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        in_chans=in_chans,
        mapper=mapper,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model





class RobustDiChaViT(nn.Module):
    def __init__(self, config: Model, **kwargs):
        super().__init__()
        self.cfg = config

        mapper = kwargs["mapper"]

        total_in_channels = len(config.in_channel_names)

        if config.pretrained_model_name == "distill":
            model = channelvit_distill(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        elif config.pretrained_model_name == "tiny":
            model = channelvit_tiny(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        elif config.pretrained_model_name == "base":
            model = channelvit_base(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        elif config.pretrained_model_name == "small":
            model = channelvit_small(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        else:
            raise ValueError("Unknown model name")

        self.feature_extractor = model
        self.classifer_head = nn.Identity()

        if "Allen" not in mapper:  ## if not Morphem dataset
            ## append an classifier layer to the model
            self.classifer_head = nn.Linear(model.num_features, config.num_classes)

        num_proxies = config.num_classes  ## depends on the number of classes of the dataset
        self.dim = model.norm.weight.shape[0]
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

        self.adaptive_interface = nn.ParameterList([self.proxies])
        
        # NEW: Channel adaptation layer for improved novel channel handling
        if getattr(config, 'use_channel_adaptation', False):
            self.channel_adaptation = nn.Sequential(
                nn.Linear(model.num_features, model.num_features),
                nn.LayerNorm(model.num_features),
                nn.GELU(),
                nn.Linear(model.num_features, model.num_features),
            )
        else:
            self.channel_adaptation = nn.Identity()

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        chunk_name: str,
        training_chunks: Optional[str] = None,
        init_first_layer=None,
        new_channel_init: Optional[NewChannelLeaveOneOut] = None,
        **kwargs,
    ) -> torch.Tensor:
        # init_first_layer: not used
        if getattr(self.cfg, 'store_attention_maps', False):
            x, ortho_proxy_loss, attention_maps = self.feature_extractor(
                x, chunk_name, training_chunks, new_channel_init)
            
            # Apply additional channel adaptation if configured
            if isinstance(self.channel_adaptation, nn.Sequential):
                x = self.channel_adaptation(x)
                
            x = self.classifer_head(x)
            
            if self.training:
                if isinstance(ortho_proxy_loss, int) and ortho_proxy_loss == 0:
                    ortho_proxy_loss = torch.tensor(0.0, device=x.device)
                return x, ortho_proxy_loss, attention_maps
            else:
                return x, attention_maps
        else:
            x, ortho_proxy_loss = self.feature_extractor(x, chunk_name, training_chunks, new_channel_init)
            
            # Apply additional channel adaptation if configured
            if isinstance(self.channel_adaptation, nn.Sequential):
                x = self.channel_adaptation(x)
                
            x = self.classifer_head(x)
            
            if self.training:
                if isinstance(ortho_proxy_loss, int) and ortho_proxy_loss == 0:
                    ortho_proxy_loss = torch.tensor(0.0, device=x.device)
                return x, ortho_proxy_loss
            else:
                return x
    
    def channel_banks_summary(self):
        """
        Returns summary statistics about the channel banks
        """
        if hasattr(self.feature_extractor.patch_embed, 'channel_features_bank') and \
           self.feature_extractor.patch_embed.channel_features_bank is not None:
            return {
                'num_channels_in_bank': self.feature_extractor.patch_embed.channel_features_bank.shape[0],
                'feature_shape': tuple(self.feature_extractor.patch_embed.channel_features_bank.shape[1:])
            }
        return {'num_channels_in_bank': 0}
    
DiChaViT = RobustDiChaViT
  
def robust_dichavit(cfg: Model, **kwargs) -> RobustDiChaViT:
    return RobustDiChaViT(config=cfg, **kwargs)


# Keep the original for backward compatibility
def dichavit(cfg: Model, **kwargs) -> RobustDiChaViT:
    print("Using robust_dichavit implementation for enhanced novel channel handling")
    return robust_dichavit(cfg, **kwargs)

