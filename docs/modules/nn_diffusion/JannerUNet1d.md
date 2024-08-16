---
layout: default
title: JannerUNet1d
nav_order: 3.2.6
parent: nn_diffusion
grand_parent: MODULES
---

# **JannerUNet1d**

> **CLASS** cleandiffuser.nn_diffusion.JannerUNet1d(in_dim: int, model_dim: int = 32, emb_dim: int = 32, kernel_size: int = 3, dim_mult: List[int] = [1, 2, 2, 2], norm_type: str = "groupnorm", attention: bool = False, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/jannerunet.py)

A modified U-Net architecture to process 1D data, proposed in [Diffuser](https://arxiv.org/abs/2205.09991). It generates trajectory sequences. One important feature is that it can handle variable-length input sequences, i.e., the sequence length during training and inference can be different.

**Parameters:**
- **in_dim** (int): The dimension of the input tensor. For state-action trajectory sequences, it is the sum of the state and action dimensions.
- **model_dim** (int): The dimension of the model. Default is 32.
- **emb_dim** (int): The dimension of the time embedding. Default is 32.
- **kernel_size** (int): The kernel size of the convolutional layers. Default is 3.
- **dim_mult** (List[int]): The multiplier for the number of channels in each layer. Default is [1, 2, 2, 2].
- **norm_type** (str): The type of normalization layer. It can be either "groupnorm" or "layernorm". If it is not one of them, no normalization layer will be used. Default is "groupnorm".
- **attention** (bool): Whether to use attention layers. Default is False.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(..., seq_len, in_dim)`.
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(..., 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(..., emb_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., seq_len, in_dim)`.
