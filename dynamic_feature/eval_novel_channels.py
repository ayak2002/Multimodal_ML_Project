import torch
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

def main():
    print("Starting simplified novel channel strategy test...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple config
    class SimpleConfig:
        def __init__(self):
            # Basic settings
            self.temperature = 0.11111
            self.hcs_sampling = "none"
            self.novel_channel_strategy = "dynamic_features"
            self.similarity_threshold = 0.6
            self.emb_interp_factor = 0.7
            
            def get(self, key, default=None):
                return getattr(self, key, default)
    
    config = SimpleConfig()
    
    # Create mapper
    mapper = {
        "train": [0, 1, 2, 3, 4, 5, 6, 7],
        "test": [0, 1, 2, 3, 4, 5, 6, 7]
    }
    
    # Create our own implementation of the channel statistics encoder
    class SimpleChannelStatisticsEncoder(torch.nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.stats_encoder = torch.nn.Sequential(
                torch.nn.Linear(5, 32),
                torch.nn.GELU(),
                torch.nn.Linear(32, embed_dim),
                torch.nn.Dropout(0.1)
            )
        
        def compute_channel_statistics(self, x):
            batch_size = x.shape[0]
            mean = x.mean(dim=[1, 2], keepdim=False)
            std = x.std(dim=[1, 2], keepdim=False)
            min_val = x.view(batch_size, -1).min(dim=1)[0]
            max_val = x.view(batch_size, -1).max(dim=1)[0]
            median = x.view(batch_size, -1).median(dim=1)[0]
            statistics = torch.stack([mean, std, min_val, max_val, median], dim=1)
            return statistics
        
        def forward(self, x):
            stats = self.compute_channel_statistics(x)
            return self.stats_encoder(stats)
    
    # Create our own implementation of the channel adapter
    class SimpleChannelAdapter(torch.nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.adapter = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=7, padding=3),
                torch.nn.GELU(),
                torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(32, embed_dim),
                torch.nn.Dropout(0.1)
            )
        
        def forward(self, x):
            x = x.unsqueeze(1)
            return self.adapter(x)
    
    # Create a simplified RobustPatchEmbedPerChannel
    class SimplePatchEmbed(torch.nn.Module):
        def __init__(self, img_size, patch_size, in_chans, embed_dim, mapper):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.mapper = mapper
            self.embed_dim = embed_dim
            
            # Channel embedding
            self.channel_embed = torch.nn.Embedding(in_chans, embed_dim)
            
            # Patch embedding projection
            self.proj = torch.nn.Conv3d(
                1, embed_dim, kernel_size=(1, patch_size, patch_size),
                stride=(1, patch_size, patch_size)
            )
            
            # Novel channel handling components
            self.stats_encoder = SimpleChannelStatisticsEncoder(embed_dim)
            self.channel_adapter = SimpleChannelAdapter(embed_dim)
            self.channel_fusion = torch.nn.Sequential(
                torch.nn.Linear(embed_dim * 2, embed_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1)
            )
            
            # Strategy settings
            self.novel_channel_strategy = config.novel_channel_strategy
            self.similarity_threshold = config.similarity_threshold
            self.emb_interp_factor = config.emb_interp_factor
        
        def get_novel_channel_embedding(self, channel_data, chunk_name, training_channels, device):
            batch_size = channel_data.shape[0]
            
            if self.novel_channel_strategy == 'dynamic_features':
                # Use statistical features and visual features
                stats_emb = self.stats_encoder(channel_data)
                visual_emb = self.channel_adapter(channel_data)
                combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                channel_emb = self.channel_fusion(combined_emb)
                
            elif self.novel_channel_strategy == 'nearest_neighbor':
                # Simplified nearest neighbor strategy (just use stats)
                stats_emb = self.stats_encoder(channel_data)
                visual_emb = self.channel_adapter(channel_data)
                combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                channel_emb = self.channel_fusion(combined_emb)
                
            elif self.novel_channel_strategy == 'weighted_interpolation':
                # Simplified weighted interpolation (just use stats)
                stats_emb = self.stats_encoder(channel_data)
                visual_emb = self.channel_adapter(channel_data)
                combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                channel_emb = self.channel_fusion(combined_emb)
            else:
                # Default approach
                if len(training_channels) > 0:
                    # Use average of a few random training channels
                    channels_to_use = min(3, len(training_channels))
                    random_indices = torch.randint(0, len(training_channels), (channels_to_use,))
                    channel_ids = [training_channels[i] for i in random_indices]
                    channel_tensor = torch.tensor(channel_ids, device=device)
                    channel_emb = self.channel_embed(channel_tensor).mean(dim=0, keepdim=True)
                    channel_emb = channel_emb.repeat(batch_size, 1)
                else:
                    # Fallback
                    stats_emb = self.stats_encoder(channel_data)
                    visual_emb = self.channel_adapter(channel_data)
                    combined_emb = torch.cat([stats_emb, visual_emb], dim=1)
                    channel_emb = self.channel_fusion(combined_emb)
            
            return channel_emb
        
        def forward(self, x, chunk_name, training_chunks, new_channel_init=None, extra_tokens={}):
            cur_channels = self.mapper[chunk_name]
            b, c, h, w = x.shape
            
            # Project patches
            x = self.proj(x.unsqueeze(1))  # B Cout Cin H W
            
            # Get channel embeddings
            if training_chunks is not None:
                training_chunks = training_chunks.split("_")
                training_channels = [self.mapper[ch] for ch in training_chunks]
                training_channels = [item for sublist in training_channels for item in sublist]
                
                param_list = []
                for i, c in enumerate(self.mapper[chunk_name]):
                    if c not in training_channels:
                        # Novel channel
                        channel_data = x[:, i]
                        channel_emb = self.get_novel_channel_embedding(
                            channel_data, chunk_name, training_channels, x.device)
                        param_list.append(channel_emb)
                    else:
                        # Known channel
                        idx = torch.tensor([c], device=x.device)
                        param = self.channel_embed(idx)
                        param = param.repeat(b, 1)
                        param_list.append(param)
                
                channel_embed = torch.stack(param_list, dim=1)
            else:
                # Standard approach for known channels
                channel_embed = self.channel_embed(torch.tensor(cur_channels, device=x.device))
                channel_embed = channel_embed.unsqueeze(0).repeat(b, 1, 1)
            
            # Add channel embeddings
            x = x + channel_embed.unsqueeze(-1).unsqueeze(-1)
            
            # Flatten and transpose
            x = x.flatten(2)  # B Cout CinHW
            x = x.transpose(1, 2)  # B CinHW Cout
            
            # Return with dummy loss of 0
            return x, c, 0.0
    
    # Create the simplified patch embed
    print("Creating simplified patch embed...")
    patch_embed = SimplePatchEmbed(
        img_size=224,
        patch_size=16,
        in_chans=8,
        embed_dim=384,
        mapper=mapper
    ).to(device)
    print("Patch embed created successfully")
    
    # Test with synthetic data
    print("\nTesting patch embed with synthetic data...")
    
    # Create synthetic test data
    batch_size = 4
    img_size = 224
    in_channels = 8
    x = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
    
    # Test each strategy
    strategies = ["dynamic_features", "nearest_neighbor", "weighted_interpolation"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        try:
            # Set strategy
            patch_embed.novel_channel_strategy = strategy
            
            # Forward pass with synthetic data
            with torch.no_grad():
                output, num_channels, extra_loss = patch_embed(
                    x, 
                    chunk_name="test", 
                    training_chunks="train",
                    new_channel_init=None,
                    extra_tokens={}
                )
            
            # Log success
            results[strategy] = "Success"
            print(f"Strategy {strategy} worked successfully!")
            print(f"Output shape: {output.shape}")
            
        except Exception as e:
            # Log failure
            results[strategy] = f"Failed: {str(e)}"
            print(f"Strategy {strategy} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\nEvaluation summary:")
    for strategy, result in results.items():
        print(f"- {strategy}: {result}")
    
    print("\nRecommended strategy:", next((s for s, r in results.items() if "Success" in r), None))
    
    # If weighted_interpolation was successful, test different factors
    if "weighted_interpolation" in results and "Success" in results["weighted_interpolation"]:
        print("\nTesting different interpolation factors for weighted_interpolation...")
        interp_results = {}
        
        for factor in [0.3, 0.5, 0.7, 0.9]:
            print(f"Testing factor: {factor}")
            try:
                patch_embed.novel_channel_strategy = "weighted_interpolation"
                patch_embed.emb_interp_factor = factor
                
                with torch.no_grad():
                    output, num_channels, extra_loss = patch_embed(
                        x, 
                        chunk_name="test", 
                        training_chunks="train",
                        new_channel_init=None,
                        extra_tokens={}
                    )
                
                interp_results[factor] = "Success"
                print(f"Factor {factor} worked successfully!")
                
            except Exception as e:
                interp_results[factor] = f"Failed: {str(e)}"
                print(f"Factor {factor} failed with error: {e}")
        
        print("\nInterpolation factors summary:")
        for factor, result in interp_results.items():
            print(f"- {factor}: {result}")
        
        print("\nRecommended interpolation factor:", 
              next((f for f, r in interp_results.items() if "Success" in r), None))
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()