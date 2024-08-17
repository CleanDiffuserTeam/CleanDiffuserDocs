---
layout: default
title: ResNet18ImageCondition
nav_order: 3.3.6
parent: nn_condition
grand_parent: MODULES
---

# **ResNet18ImageCondition**

> **CLASS** cleandiffuser.nn_diffusion.ResNet18ImageCondition(image_sz: int, in_channel: int, emb_dim: int, act_fn: Callable[[], nn.Module] = nn.ReLU, use_group_norm: bool = True, group_channels: int = 16, use_spatial_softmax: bool = True, dropout: float = 0.0) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/resnets.py)

A ResNet18 for image condition. It encodes the input image into a fixed-size embedding. The implementation is adapted from [DiffusionPolicy](https://diffusion-policy.cs.columbia.edu/). Compared to the original ResNet18, we replace `BatchNorm2d` with `GroupNorm`, and use a SpatialSoftmax instead of an average pooling layer.


**Parameters:**
- **image_sz** (int): Size of the input image. The image is assumed to be square.
- **in_channel** (int): Number of input channels. 3 for RGB images.
- **emb_dim** (int): Dimension of the output embedding.
- **act_fn** (Callable[[], nn.Module]): Activation function to use in the network. Default is ReLU.
- **use_group_norm** (bool): Whether to use GroupNorm instead of BatchNorm. Default is True.
- **group_channels** (int): Number of channels per group in GroupNorm. Default is 16.
- **use_spatial_softmax** (bool): Whether to use SpatialSoftmax instead of average pooling. Default is True.
- **dropout** (float): Condition Dropout rate. Default is 0.0.

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(b, n, c, h, w)` or `(b, c, h, w)`. Here, `n` is sequence length. `(c, h, w)` is the shape of the image.
- **mask** (Optional[torch.Tensor]): The mask tensor. Default is None. None means no mask.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, n, emb_dim)` or `(b, emb_dim)`. Each element in the batch has a probability of `dropout` to be zeros.

