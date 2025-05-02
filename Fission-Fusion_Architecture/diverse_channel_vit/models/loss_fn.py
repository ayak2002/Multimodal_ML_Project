from torch import nn, Tensor
import torch.nn.functional as F
import torch
from utils import pairwise_distance_v2


def proxy_loss(proxies, img_emb, gt_imgs, scale: float | nn.Parameter) -> Tensor:
    """
    proxies: shape of (num_classes, dim)
    img_emb: shape of (num_imgs, dim)
    gt_imgs: shape of (num_imgs)
    """
    proxies_emb = scale * F.normalize(proxies, p=2, dim=-1)
    img_emb = scale * F.normalize(img_emb, p=2, dim=-1)

    img_dist = pairwise_distance_v2(proxies=proxies_emb, x=img_emb, squared=True)
    img_dist = img_dist * -1.0

    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    img_loss = cross_entropy(img_dist, gt_imgs)
    return img_loss


def ortho_proj_loss_fn_v2(features, labels, gamma_s, gamma_d, reverse_pos_pairs: bool, use_square: bool):
    """
    features: shape (b, num_tokens, d)
    labels: shape (num_tokens)
    gamma_s, gamma_d: lambda_s and lambda_d in E.q (2) and (3) in the paper
    reverse_pos_pairs: If true, we want each token to be orthogonal to all other tokens, regarless of their channels.
    """
    device = features.device
    #  features are normalized
    features = F.normalize(features, p=2, dim=-1)

    labels = labels[None, :, None]  # extend dims

    mask = torch.eq(labels, labels.transpose(-2, -1)).bool().to(device)
    eye = torch.eye(mask.shape[-2], mask.shape[-1]).bool().to(device).unsqueeze(0)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    dot_prod = torch.matmul(features, features.transpose(-2, -1))

    mask_pos_sum = mask_pos.sum(dim=(-2, -1)) + 1e-6
    mask_neg_sum = mask_neg.sum(dim=(-2, -1)) + 1e-6

    pos_pairs_mean = (mask_pos * dot_prod).sum(dim=(-2, -1)) / mask_pos_sum
    neg_pairs_mean = (mask_neg * dot_prod).sum(dim=(-2, -1)) / mask_neg_sum

    if use_square:
        neg_pairs_mean = neg_pairs_mean**2

    if reverse_pos_pairs:
        if use_square:
            pos_pairs_mean = pos_pairs_mean**2
        loss = gamma_s * pos_pairs_mean + gamma_d * neg_pairs_mean
    else:
        loss = gamma_s * (1.0 - pos_pairs_mean) + gamma_d * neg_pairs_mean
    return loss.mean()


def feature_separation_loss(shared_features, specific_features):
    """
    Compute cosine similarity between shared and channel-specific features to ensure they learn different information.
    
    Args:
        shared_features: Tensor of shape (batch_size, num_tokens, shared_dim)
        specific_features: Tensor of shape (batch_size, num_tokens, specific_dim)
        
    Returns:
        Loss value encouraging orthogonality between shared and specific features
    """
    # Normalize features
    shared_norm = F.normalize(shared_features, p=2, dim=-1)
    specific_norm = F.normalize(specific_features, p=2, dim=-1)
    
    # Compute cosine similarity for each token position
    batch_size, num_tokens, _ = shared_features.shape
    
    # Handle different dimensions by using a simpler approach
    if shared_features.shape[-1] != specific_features.shape[-1]:
        # Use the smaller dimension as the common dimension
        min_dim = min(shared_features.shape[-1], specific_features.shape[-1])
        
        # Truncate to common dimension
        shared_flat = shared_norm[..., :min_dim].reshape(batch_size * num_tokens, -1)
        specific_flat = specific_norm[..., :min_dim].reshape(batch_size * num_tokens, -1)
    else:
        # If dimensions match, no truncation needed
        shared_flat = shared_norm.reshape(batch_size * num_tokens, -1)
        specific_flat = specific_norm.reshape(batch_size * num_tokens, -1)
    
    # Compute cosine similarity
    similarity = torch.sum(shared_flat * specific_flat, dim=1)
    
    # Use absolute value of similarity to penalize both positive and negative correlations
    # This is more sensitive than squaring for small values
    loss = torch.mean(torch.abs(similarity))
    
    return loss


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation"
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class ChannelClassifier(nn.Module):
    """
    Simple classifier to predict channel identity from features.
    Used for adversarial training to ensure shared features are channel-agnostic.
    Includes a gradient reversal layer for proper adversarial training.
    """
    def __init__(self, feature_dim, num_channels, hidden_dim=None, alpha=1.0):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 2
        
        # Add gradient reversal layer
        self.grad_reverse = GradientReversal(alpha)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels)
        )
    
    def forward(self, x):
        # x: (batch, tokens, feature_dim)
        # We use the CLS token (first token) for classification
        cls_token = x[:, 0]
        # Apply gradient reversal before classification
        reversed_features = self.grad_reverse(cls_token)
        return self.classifier(reversed_features)


def adversarial_channel_loss(shared_features, channel_labels, channel_classifier, scaling_factor=1.0):
    """
    Compute adversarial loss to make shared features channel-agnostic.
    Uses gradient reversal for proper adversarial training.
    
    Args:
        shared_features: Tensor of shape (batch_size, num_tokens, feature_dim)
        channel_labels: Tensor of shape (batch_size) containing channel indices
        channel_classifier: ChannelClassifier module with gradient reversal
        scaling_factor: Factor to scale the KL divergence loss (default: 1.0)
        
    Returns:
        Loss value encouraging shared features to be channel-agnostic
    """
    # Get channel predictions from classifier (gradient reversal happens inside the classifier)
    channel_logits = channel_classifier(shared_features)
    batch_size = channel_logits.size(0)
    num_classes = channel_logits.size(1)
    
    # Create a uniform target distribution
    uniform_target = torch.ones_like(channel_logits) / num_classes
    
    # Use KL divergence to measure how far the predictions are from uniform
    log_softmax = F.log_softmax(channel_logits, dim=1)
    
    # Calculate KL divergence
    kl_div = F.kl_div(log_softmax, uniform_target, reduction='batchmean')
    
    # Scale the loss by the provided scaling factor
    scaled_loss = kl_div * scaling_factor
    
    # With gradient reversal, we can just return the loss directly
    # The gradients will be automatically reversed during backpropagation
    return scaled_loss
