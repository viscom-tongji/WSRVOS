from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=tensor.dtype)
    numerator = (tensor * mask.unsqueeze(-1)).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    while denom.ndim < numerator.ndim:
        denom = denom.unsqueeze(-1)
    return numerator / denom


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid()
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (numerator + 1.0) / (denominator + 1.0)).mean()


def token_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    intersection = (mask_a & mask_b).float().sum(dim=-1)
    union = (mask_a | mask_b).float().sum(dim=-1).clamp_min(1.0)
    return intersection / union


def temporal_ranking_loss(pseudo_masks: torch.Tensor, epsilon: float) -> torch.Tensor:
    batch_size, num_frames = pseudo_masks.shape[:2]
    if num_frames < 3:
        return pseudo_masks.new_zeros(())
    losses = []
    for t in range(num_frames - 2):
        base = pseudo_masks[:, t]
        for l in range(t + 1, num_frames - 1):
            iou_near = token_iou(base, pseudo_masks[:, l])
            for n in range(l + 1, num_frames):
                iou_far = token_iou(base, pseudo_masks[:, n])
                losses.append(F.relu(iou_far - iou_near - epsilon))
    if not losses:
        return pseudo_masks.new_zeros(())
    return torch.stack(losses, dim=0).mean()


def classification_loss(
    frame_scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    frame_scores = frame_scores.clamp(1e-6, 1.0 - 1e-6)
    sample_scores = frame_scores.mean(dim=1)
    losses = F.binary_cross_entropy(sample_scores, labels, reduction="none")
    losses = losses * valid_mask.float()
    return losses.sum() / valid_mask.float().sum().clamp_min(1.0)


def segmentation_loss(
    positive_logits: torch.Tensor,
    pseudo_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_tokens, num_positive = positive_logits.shape
    logits = positive_logits.permute(0, 1, 3, 2).reshape(batch_size * num_frames * num_positive, num_tokens)
    pseudo = pseudo_masks.unsqueeze(2).expand(-1, -1, num_positive, -1).reshape(batch_size * num_frames * num_positive, num_tokens)
    return binary_focal_loss(logits, pseudo), dice_loss(logits, pseudo)
