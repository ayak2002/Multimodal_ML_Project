import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dichavit import FissionModule, FusionModule, ChannelVisionTransformer
from models.loss_fn import feature_separation_loss, adversarial_channel_loss, ChannelClassifier

def test_standalone_modules():
    """Test the standalone FissionModule and FusionModule."""
    print("=== Testing Standalone Modules ===")
    
    # Create sample input
    batch_size = 2
    seq_len = 8
    feature_dim = 16
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # Test FissionModule
    fission_module = FissionModule(
        feature_dim=feature_dim,
        hidden_dim=feature_dim*2,
        shared_dim=10,
        specific_dim=6
    )
    
    shared, specific = fission_module(x)
    print(f"FissionModule output shapes:")
    print(f"  shared: {shared.shape}")
    print(f"  specific: {specific.shape}")
    
    # Test FusionModule
    fusion_module = FusionModule(
        shared_dim=10,
        specific_dim=6,
        fusion_hidden_dim=20,
        fusion_out_dim=12
    )
    
    fused = fusion_module(shared, specific)
    print(f"FusionModule output shape: {fused.shape}")
    print(f"Sample fused output: {fused[0, 0]}")
    
    # Test feature separation loss
    sep_loss = feature_separation_loss(shared, specific)
    print(f"Feature separation loss: {sep_loss.item():.4f}")
    
    # Test adversarial channel loss
    num_channels = 3
    channel_classifier = ChannelClassifier(10, num_channels)
    channel_labels = torch.randint(0, num_channels, (batch_size,))
    adv_loss = adversarial_channel_loss(shared, channel_labels, channel_classifier)
    print(f"Adversarial channel loss: {adv_loss.item():.4f}")

def test_integrated_fission():
    print("\n=== Testing Integrated FissionModule in DiChaViT ===")
    
    # Import here to avoid circular imports
    from models.dichavit import ChannelVisionTransformer
    
    # Create a minimal config class with the necessary attributes
    class DummyConfig:
        def __init__(self, use_fission=False, use_parallel=False, use_fusion=False):
            self.use_fission_module = use_fission
            self.use_parallel_paths = use_parallel
            self.use_fusion_module = use_fusion
            self.share_transformer_weights = True
            self.fission_hidden_dim = 0
            self.fission_shared_dim = 0
            self.fission_specific_dim = 0
            self.fusion_hidden_dim = 0
            self.fusion_out_dim = 0
            self.dropout_tokens_hcs = None
            self.block_type = "block"
            self.drop_path_rate = 0.0
            # Additional required attributes
            self.temperature = 0.1
            self.proxy_loss_lambda = 0
            self.ortho_loss_v1_lambda = 0
            self.hcs_sampling = "none"  
            self.orthogonal_channel_emb_init = False
            self.freeze_channel_emb = False
            self.proxy_orthogonal_init = False
            self.hcs_sampling_temp = 0.1
        
        # Add getter method to mimic hydra config behavior
        def get(self, key, default=None):
            return getattr(self, key, default)
        
    # Test parameters
    batch_size = 2
    img_size = 32
    patch_size = 16
    in_chans = 3
    embed_dim = 96
    
    # Create dummy image and channel mapping
    x = torch.randn(batch_size, in_chans, img_size, img_size)
    mapper = {"test_chunk": [0, 1, 2]}  # Dummy channel mapping
    
    # Test 1: Legacy mode (no fission, no parallel paths)
    print("Testing with use_fission_module=False (legacy behavior):")
    config_legacy = DummyConfig(use_fission=False, use_parallel=False, use_fusion=False)
    model_legacy = ChannelVisionTransformer(
        config=config_legacy,
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans,
        mapper=mapper,
        embed_dim=embed_dim
    )
    
    # Run forward pass with legacy config
    try:
        output_legacy = model_legacy(x, chunk_name="test_chunk")
        if isinstance(output_legacy, tuple):
            print(f"Legacy output is a tuple with {len(output_legacy)} elements")
            for i, item in enumerate(output_legacy):
                if hasattr(item, 'shape'):
                    print(f"  Element {i} shape: {item.shape}")
                else:
                    print(f"  Element {i} type: {type(item)}")
        else:
            print(f"Legacy output shape: {output_legacy.shape}")
        print("Legacy test passed - no FissionModule output printed")
    except Exception as e:
        print(f"Legacy test failed with error: {e}")
    
    # Test 2: Fission only (no parallel paths)
    print("\nTesting with use_fission_module=True, use_parallel_paths=False:")
    config_fission = DummyConfig(use_fission=True, use_parallel=False, use_fusion=False)
    model_fission = ChannelVisionTransformer(
        config=config_fission,
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans,
        mapper=mapper,
        embed_dim=embed_dim
    )
    
    # Run forward pass with fission enabled
    try:
        output_fission = model_fission(x, chunk_name="test_chunk")
        if isinstance(output_fission, tuple):
            print(f"Fission output is a tuple with {len(output_fission)} elements")
            for i, item in enumerate(output_fission):
                if hasattr(item, 'shape'):
                    print(f"  Element {i} shape: {item.shape}")
                else:
                    print(f"  Element {i} type: {type(item)}")
        else:
            print(f"Fission output shape: {output_fission.shape}")
        print("Fission test passed - should see FissionModule output printed above")
    except Exception as e:
        print(f"Fission test failed with error: {e}")
    
    # Test 3: Fission with parallel paths
    print("\nTesting with use_fission_module=True, use_parallel_paths=True:")
    config_parallel = DummyConfig(use_fission=True, use_parallel=True, use_fusion=False)
    model_parallel = ChannelVisionTransformer(
        config=config_parallel,
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans,
        mapper=mapper,
        embed_dim=embed_dim
    )
    
    # Run forward pass with parallel paths enabled
    try:
        output_parallel = model_parallel(x, chunk_name="test_chunk")
        if isinstance(output_parallel, tuple):
            print(f"Parallel output is a tuple with {len(output_parallel)} elements")
            for i, item in enumerate(output_parallel):
                if hasattr(item, 'shape'):
                    print(f"  Element {i} shape: {item.shape}")
                else:
                    print(f"  Element {i} type: {type(item)}")
        else:
            print(f"Parallel output shape: {output_parallel.shape}")
        print("Parallel test passed - should see both FissionModule and ParallelPaths output printed above")
    except Exception as e:
        print(f"Parallel test failed with error: {e}")
    
    # Test 4: Fission with parallel paths and fusion
    print("\nTesting with use_fission_module=True, use_parallel_paths=True, use_fusion_module=True:")
    config_fusion = DummyConfig(use_fission=True, use_parallel=True, use_fusion=True)
    model_fusion = ChannelVisionTransformer(
        config=config_fusion,
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans,
        mapper=mapper,
        embed_dim=embed_dim
    )
    
    # Run forward pass with fusion enabled
    try:
        output_fusion = model_fusion(x, chunk_name="test_chunk")
        if isinstance(output_fusion, tuple):
            print(f"Fusion output is a tuple with {len(output_fusion)} elements")
            for i, item in enumerate(output_fusion):
                if hasattr(item, 'shape'):
                    print(f"  Element {i} shape: {item.shape}")
                else:
                    print(f"  Element {i} type: {type(item)}")
        else:
            print(f"Fusion output shape: {output_fusion.shape}")
        print("Fusion test passed - should see FissionModule, ParallelPaths, and FusionModule output printed above")
    except Exception as e:
        print(f"Fusion test failed with error: {e}")

if __name__ == "__main__":
    # Test standalone modules first
    test_standalone_modules()
    
    # Test integrated fission module
    test_integrated_fission()
