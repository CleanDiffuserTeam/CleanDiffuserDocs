---
layout: default
title: EarlyConvViTMultiViewImageCondition
nav_order: 3.3.7
parent: nn_condition
grand_parent: MODULES
---

# **EarlyConvViTMultiViewImageCondition**

> **CLASS** cleandiffuser.nn_diffusion.EarlyConvViTMultiViewImageCondition(image_sz: Tuple[int], in_channels: Tuple[int], lowdim_sz: Optional[int], To: int, d_model: int = 384, nhead: int = 6, num_layers: int = 2, attn_dropout: float = 0.0, ffn_dropout: float = 0.0, patch_size: Tuple[int] = (16, 16), channels_per_group: Tuple[int] = (16, 16), kernel_sizes: Tuple[Tuple[int]] = ((3, 3, 3, 3), (3, 3, 3, 3)), strides: Tuple[Tuple[int]] = ((2, 2, 2, 2), (2, 2, 2, 2), features: Tuple[Tuple[int]] = ((32, 64, 128, 256), (32, 64, 128, 256)), padding: Tuple[Tuple[int]] = ((1, 1, 1, 1), (1, 1, 1, 1)))) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/early_conv_vit.py)

An early-CNN Vision Transformer (ViT) for multi-view image condition. This architecture is proposed in [Early Convolutional Vision Transformer](https://arxiv.org/pdf/2106.14881) and demonstrated to be effective for CV tasks. The vision encoder in [Octo](https://arxiv.org/pdf/2405.12213) is mainly based on this architecture. Each view is processed by a separate CNN and the resulting tokens are concatenated along the token dimension, which are then processed by a transformer. The output of the learnable 'readout' token is returned as the final representation.

**Parameters:**
- **image_sz** (Tuple[int]): The size of the input image for each view. Assumes square images.
- **in_channels** (Tuple[int]): The number of input channels for each view.
- **lowdim_sz** (Optional[int]): The size of the low-dimensional condition. If None, no low-dimensional condition is used.
- **To** (int): The number of frames for each view.
- **d_model** (int): The dimension of the transformer token. Default is 384.
- **nhead** (int): The number of heads in the transformer. Default is 6.
- **num_layers** (int): The number of transformer layers. Default is 2.
- **attn_dropout** (float): The dropout rate for the attention layer. Default is 0.0.
- **ffn_dropout** (float): The dropout rate for the feedforward layer. Default is 0.0.
- **patch_size** (Tuple[int]): The size of the patch for each view. Default is (16, 16).
- **channels_per_group** (Tuple[int]): The number of channels per group in the CNN. Default is (16, 16).
- **kernel_sizes** (Tuple[Tuple[int]]): The kernel sizes for each CNN layer. Default is ((3, 3, 3, 3), (3, 3, 3, 3)).
- **strides** (Tuple[Tuple[int]]): The strides for each CNN layer. Default is ((2, 2, 2, 2), (2, 2, 2, 2)).
- **features** (Tuple[Tuple[int]]): The number of features for each CNN layer. Default is ((32, 64, 128, 256), (32, 64, 128, 256)).
- **padding** (Tuple[Tuple[int]]): The padding for each CNN layer. Default is ((1, 1, 1, 1), (1, 1, 1, 1)).

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(b, v, n, c, h, w)`. Here, `v` is the number of different views, `n` is sequence length, and (c, h, w)` is the shape of the image.
- **mask** (Optional[torch.Tensor]): The mask tensor. Default is None. None means no mask.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, v, n, emb_dim)`. Each element in the batch has a probability of `dropout` to be zeros.
