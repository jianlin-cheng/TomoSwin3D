# Code for Dice loss and Weighted Binary Cross-Entropy Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(predictions, targets, class_weights=None, weighted=False, 
               alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
    """
    Focal Loss for multi-class segmentation with extreme class imbalance.
    
    Focal Loss down-weights the loss assigned to well-classified examples,
    allowing the model to focus more on hard, misclassified examples.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where:
        p_t = probability of the true class
        α_t = class weight for balancing class frequencies
        γ (gamma) = focusing parameter for modulating loss (γ >= 0)
    
    Args:
        predictions (torch.Tensor): Model logits of shape (B, C, D, H, W)
                                   where C is number of classes (e.g., 130)
        targets (torch.Tensor): Ground truth labels of shape (B, D, H, W)
                               with values in range [0, C-1]
        class_weights (torch.Tensor, optional): Per-class weights of shape (C,)
                                               for addressing class imbalance
        weighted (bool): Whether to apply class weights. Default: False
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples.
                      Can be a scalar or per-class tensor. Default: 0.25
        gamma (float): Focusing parameter γ >= 0. 
                      - γ=0: Focal Loss = Cross Entropy Loss
                      - γ=2: Recommended default (down-weights easy examples by 4x)
                      - γ=5: Very strong focus on hard examples
                      Higher gamma = more focus on hard-to-classify examples.
        reduction (str): Specifies reduction to apply to output.
                        Options: 'mean', 'sum', 'none'. Default: 'mean'
        eps (float): Small constant for numerical stability. Default: 1e-8
    
    Returns:
        torch.Tensor: Computed focal loss (scalar if reduction='mean'/'sum')
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        https://arxiv.org/abs/1708.02002
    
    Example:
        >>> predictions = torch.randn(2, 130, 64, 64, 64)  # 2 samples, 130 classes
        >>> targets = torch.randint(0, 130, (2, 64, 64, 64))
        >>> loss = focal_loss(predictions, targets, gamma=2.0)
    """
    # Input validation
    B, C, D, H, W = predictions.shape
    assert targets.shape == (B, D, H, W), \
        f"Target shape {targets.shape} doesn't match predictions shape (B, D, H, W) = {(B, D, H, W)}"
    assert gamma >= 0, f"Gamma must be non-negative, got {gamma}"
    
    # Compute softmax probabilities: shape (B, C, D, H, W)
    probs = F.softmax(predictions, dim=1)
    
    # Clamp probabilities for numerical stability
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    
    # Gather probabilities of the ground truth class for each voxel
    # targets: (B, D, H, W) -> (B, D, H, W, 1) for gathering
    targets_long = targets.long().unsqueeze(1)  # (B, 1, D, H, W)
    
    # Gather the probability of the true class: p_t
    # probs: (B, C, D, H, W), targets_long: (B, 1, D, H, W)
    pt = torch.gather(probs, dim=1, index=targets_long)  # (B, 1, D, H, W)
    pt = pt.squeeze(1)  # (B, D, H, W)
    
    # Compute the focal term: (1 - p_t)^gamma
    # This down-weights easy examples (high p_t) and focuses on hard examples (low p_t)
    focal_weight = (1.0 - pt) ** gamma  # (B, D, H, W)
    
    # Compute cross entropy: -log(p_t)
    ce = -torch.log(pt)  # (B, D, H, W)
    
    # Combine focal weight with cross entropy
    focal = focal_weight * ce  # (B, D, H, W)
    
    # Apply alpha weighting (can be scalar or per-class)
    if isinstance(alpha, (float, int)):
        focal = alpha * focal
    elif isinstance(alpha, torch.Tensor):
        # Per-class alpha: gather alpha for each target class
        alpha_t = alpha[targets.long()]  # (B, D, H, W)
        focal = alpha_t * focal
    
    # Apply class weights if specified
    if weighted and class_weights is not None:
        # Gather class weights for each target voxel
        class_weights = class_weights.to(predictions.device)
        weight_map = class_weights[targets.long()]  # (B, D, H, W)
        focal = focal * weight_map
    
    # Apply reduction
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    elif reduction == 'none':
        return focal
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose from ['mean', 'sum', 'none']")


def dice_loss(predictions, targets, class_weights=None, weighted=False, 
              smooth=1e-6, reduction='mean'):
    """
    Optimized Dice Loss for 3D multi-class segmentation.
    
    Args:
        predictions (torch.Tensor): Model logits of shape (B, C, D, H, W)
        targets (torch.Tensor): Ground truth labels of shape (B, D, H, W)
        class_weights (torch.Tensor, optional): Per-class weights of shape (C,)
        weighted (bool): Whether to apply class weights
        smooth (float): Smoothing factor to avoid division by zero
        reduction (str): 'mean', 'sum', or 'none'
    
    Returns:
        torch.Tensor: Dice loss value
    """
    B, C, D, H, W = predictions.shape
    
    # Apply softmax to get probabilities
    probs = F.softmax(predictions, dim=1)  # (B, C, D, H, W)
    
    # Convert targets to one-hot encoding
    targets_onehot = F.one_hot(targets.long(), num_classes=C)  # (B, D, H, W, C)
    targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
    
    # Flatten spatial dimensions for easier computation
    probs_flat = probs.view(B, C, -1)  # (B, C, D*H*W)
    targets_flat = targets_onehot.view(B, C, -1)  # (B, C, D*H*W)
    
    # Compute intersection and union for each class
    intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
    union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
    
    # Compute Dice coefficient for each class
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)  # (B, C)
    
    # Convert to Dice loss (1 - Dice coefficient)
    dice_loss_per_class = 1.0 - dice_coeff  # (B, C)
    
    # Apply class weights if specified
    if weighted and class_weights is not None:
        class_weights = class_weights.to(predictions.device)
        # Expand class weights to match batch dimension
        class_weights = class_weights.unsqueeze(0).expand(B, -1)  # (B, C)
        dice_loss_per_class = dice_loss_per_class * class_weights
    
    # Apply reduction
    if reduction == 'mean':
        return dice_loss_per_class.mean()
    elif reduction == 'sum':
        return dice_loss_per_class.sum()
    elif reduction == 'none':
        return dice_loss_per_class
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def generalized_dice_loss(predictions, targets, class_weights=None, weighted=False,
                          smooth=1e-6, reduction='mean'):
    """
    Generalized Dice Loss with better handling of class imbalance.
    
    This version uses the square of the union in the denominator to better
    handle class imbalance, as proposed in the original paper.
    """
    B, C, D, H, W = predictions.shape
    
    # Apply softmax to get probabilities
    probs = F.softmax(predictions, dim=1)  # (B, C, D, H, W)
    
    # Convert targets to one-hot encoding
    targets_onehot = F.one_hot(targets.long(), num_classes=C)  # (B, D, H, W, C)
    targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
    
    # Flatten spatial dimensions
    probs_flat = probs.view(B, C, -1)  # (B, C, D*H*W)
    targets_flat = targets_onehot.view(B, C, -1)  # (B, C, D*H*W)
    
    # Compute intersection and union
    intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
    union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
    
    # Compute class weights based on inverse frequency
    if weighted and class_weights is not None:
        class_weights = class_weights.to(predictions.device)
        # Normalize class weights
        class_weights = class_weights / class_weights.sum()
    else:
        # Use inverse frequency weighting
        class_counts = targets_flat.sum(dim=(0, 2))  # (C,)
        class_weights = 1.0 / (class_counts + smooth)
        class_weights = class_weights / class_weights.sum()
    
    # Expand weights to match batch dimension
    class_weights = class_weights.unsqueeze(0).expand(B, -1)  # (B, C)
    
    # Compute weighted Dice coefficient
    weighted_intersection = (class_weights * intersection).sum(dim=1)  # (B,)
    weighted_union = (class_weights * union).sum(dim=1)  # (B,)
    
    # Compute generalized Dice loss
    dice_coeff = (2.0 * weighted_intersection + smooth) / (weighted_union + smooth)  # (B,)
    dice_loss = 1.0 - dice_coeff  # (B,)
    
    # Apply reduction
    if reduction == 'mean':
        return dice_loss.mean()
    elif reduction == 'sum':
        return dice_loss.sum()
    elif reduction == 'none':
        return dice_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def dice_ce_loss(predictions, targets, class_weights=None, weighted=False,
                 dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
    """
    Combined Dice + Cross-Entropy Loss for better training stability.
    
    Args:
        dice_weight (float): Weight for Dice loss component
        ce_weight (float): Weight for Cross-Entropy loss component
    """
    # Dice loss component
    dice = dice_loss(predictions, targets, class_weights, weighted, smooth)
    
    # Cross-entropy loss component
    ce_weight_tensor = class_weights if weighted else None
    ce = F.cross_entropy(predictions, targets, weight=ce_weight_tensor)
    
    # Combine losses
    total_loss = dice_weight * dice + ce_weight * ce
    
    return total_loss


def combined_loss(predictions, targets, class_weights, weighted, loss_function, loss_activation, 
                  alpha=0.5, focal_alpha=0.25, focal_gamma=2.0, dice_weight=0.5, ce_weight=0.5):
    """
    Updated combined loss function with corrected Dice loss implementations.
    
    Loss function options:
      - 'CE'            : (weighted) Cross-Entropy Loss
      - 'Dice'          : Standard Dice Loss
      - 'GeneralizedDice': Generalized Dice Loss (better for class imbalance)
      - 'DiceCE'        : Combined Dice + Cross-Entropy Loss
      - 'BCE'           : (weighted) Binary Cross-Entropy Loss with logits
      - 'Focal'         : Focal Loss for handling class imbalance
      - 'Combined'      : weighted combination of CE and Dice

    Args:
        predictions: logits tensor of shape (B, C, D, H, W)
        targets:      label tensor of shape (B, D, H, W)
        class_weights: 1D tensor of length C for CE weighting
        weighted:      bool, whether to apply class/pos weights
        loss_function: one of ['CE', 'Dice', 'GeneralizedDice', 'DiceCE', 'BCE', 'Focal', 'Combined']
        loss_activation: activation type used by Dice loss (legacy parameter)
        alpha:         float weight between CE and Dice in Combined mode (default: 0.5)
        focal_alpha:   float weight for Focal Loss balancing (default: 0.25)
        focal_gamma:   float focusing parameter for Focal Loss (default: 2.0)
        dice_weight:   float weight for Dice component in DiceCE mode (default: 0.5)
        ce_weight:     float weight for CE component in DiceCE mode (default: 0.5)

    Returns:
        loss: scalar tensor
    
    Example usage in train.py:
        loss = combined_loss(pred, y, class_weights=class_weights, weighted=True, 
                           loss_function='Dice', loss_activation='__')
    """
    # CROSS-ENTROPY LOSS
    if loss_function == 'CE':
        weight = class_weights if weighted else None
        loss = F.cross_entropy(predictions, targets, weight=weight, reduction='mean')

    # STANDARD DICE LOSS
    elif loss_function == 'Dice':
        loss = dice_loss(predictions, targets, class_weights, weighted, smooth=1e-6)

    # GENERALIZED DICE LOSS
    elif loss_function == 'GeneralizedDice':
        loss = generalized_dice_loss(predictions, targets, class_weights, weighted, smooth=1e-6)

    # COMBINED DICE + CE LOSS
    elif loss_function == 'DiceCE':
        loss = dice_ce_loss(predictions, targets, class_weights, weighted, 
                           dice_weight, ce_weight)

    # BINARY CROSS-ENTROPY WITH LOGITS
    elif loss_function == 'BCE':
        # NOTE: `predictions` must be logits with shape either:
        #   - (B, 2, D, H, W)  [channels-first], or
        #   - (B, D, H, W, 2)  [channels-last]
        # `F.one_hot` yields (B, D, H, W, C); align layout to match logits before BCE.
        targets_onehot = F.one_hot(targets.long(), num_classes=2).float()
        if predictions.dim() != 5:
            raise ValueError(f"BCE expects 5D logits, got predictions.shape={tuple(predictions.shape)}")

        # channels-first logits: (B, C, ...)
        if predictions.shape[1] == 2:
            targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).contiguous()
        # channels-last logits: (..., C)
        elif predictions.shape[-1] == 2:
            pass
        else:
            raise ValueError(
                "BCE expects binary logits with C=2 along channel dimension. "
                f"Got predictions.shape={tuple(predictions.shape)}"
            )

        if weighted:
            # class_weights = [w_bg, w_fg]
            w_bg, w_fg = class_weights[0], class_weights[1]
            # Upscale loss on positive (foreground) logits vs background for imbalanced data.
            pos_ratio = w_fg / (w_bg + 1e-8)
            pos_weight = torch.as_tensor(pos_ratio, device=predictions.device, dtype=predictions.dtype)

            loss = F.binary_cross_entropy_with_logits(
                input=predictions,
                target=targets_onehot,
                weight=None,
                pos_weight=pos_weight,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(predictions, targets_onehot)

    # FOCAL LOSS
    elif loss_function == 'Focal':
        loss = focal_loss(
            predictions=predictions,
            targets=targets,
            class_weights=class_weights,
            weighted=weighted,
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean'
        )

    # COMBINED CE + DICE (original)
    elif loss_function == 'Combined':
        weight = class_weights if weighted else None
        ce_loss = F.cross_entropy(predictions, targets, weight=weight, reduction='mean')
        dice_loss_val = dice_loss(predictions, targets, class_weights, weighted)
        loss = alpha * ce_loss + (1 - alpha) * dice_loss_val

    # DEFAULT: COMBINED CE + DICE
    else:
        weight = class_weights if weighted else None
        ce_loss = F.cross_entropy(predictions, targets, weight=weight, reduction='mean')
        dice_loss_val = dice_loss(predictions, targets, class_weights, weighted)
        loss = alpha * ce_loss + (1 - alpha) * dice_loss_val

    return loss
