# input: pred and label
# output: total losses

import torch
import torch.nn.functional as F


def cls_cross_entropy_loss(pred_cls, target_cls):
    """
    Args:
        pred_cls:  logits Tensor   (N, 2K, ft_size, ft_size)
        target_cls: {-1,0,1} Tensor (N, K, ft_size, ft_size)

    Returns:
        cls loss: scalar
    """
    N, KK, SIZE, SIZE = pred_cls.shape
    pred_cls = pred_cls.view(N, 2, -1, SIZE, SIZE)  # (N, 2, K, ft_size, ft_size)
    pred_cls = pred_cls.permute(0, 2, 3, 4, 1).contiguous()  # (N, K, ft_size, ft_size, 2)
    neg_idx = torch.where(target_cls == 0)
    pos_idx = torch.where(target_cls == 1)
    target_pos = target_cls[pos_idx]  # (#pos,)
    target_neg = target_cls[neg_idx]  # (#neg,)
    pred_pos = pred_cls[pos_idx]      # (#pos, 2)
    pred_neg = pred_cls[neg_idx]      # (#pos, 2)
    cls_pos_loss = F.cross_entropy(input=pred_pos, target=target_pos)
    cls_neg_loss = F.cross_entropy(input=pred_neg, target=target_neg)
    return 0.5 * cls_pos_loss + 0.5 * cls_neg_loss

def reg_smooth_l1_loss(pred_reg, target_reg, mask):
    """
    Args:
        pred_reg: Tensor   (N, 4K, ft_size, ft_size)
        target_reg: Tensor (N, 4, K, ft_size, ft_size)
        mask:   Tensor   (N, K, ft_size, ft_size)
        pos_num: number of positive anchors (N,)

    Returns:
        cls loss: scalar
    """
    N, _, SIZE, SIZE = pred_reg.shape
    pred_reg = pred_reg.view(N, 4, -1 , SIZE, SIZE)  # (N, 4, K, ft_size, ft_size)
    diff = F.smooth_l1_loss(pred_reg, target_reg, reduction='none')  # won't change shape
    diff = diff.sum(dim=1).view(N, -1, SIZE, SIZE)
    loss = diff * mask
    return loss.sum().div(N)




