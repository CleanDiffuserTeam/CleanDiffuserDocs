---
layout: default
title: ChiUNet1d
nav_order: 3.2.7
parent: nn_diffusion
grand_parent: MODULES
---

# **ChiUNet1d**

> **CLASS** cleandiffuser.nn_diffusion.ChiUNet1d(act_dim: int, obs_dim: int, To: int, model_dim: int = 256, emb_dim: int = 256, kernel_size: int = 5, cond_predict_scale: bool = True, obs_as_global_cond: bool = True, dim_mult: List[int] = [1, 2, 2, 2], timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/chiunet.py)

A modified U-Net architecture to process 1D data, proposed in [DiffusionPolicy](https://diffusion-policy.cs.columbia.edu/). Compared to the original U-Net, it incorporates FiLM layers to condition the model on the observation sequence. It generates action trajectory sequences. One important feature is that it can handle variable-length input sequences, i.e., the sequence length during training and inference can be different.

**Parameters:**
- **act_dim** (int): The dimension of the action tensor $$\bm x_t$$.
- **obs_dim** (int): The dimension of the observation tensor $$\bm c$$.
- **To** (int): The number of observations to consider. 1 means only the current observation.
- **model_dim** (int): The dimension of the model. Default is 256.
- **emb_dim** (int): The dimension of the time embedding. Default is 256.
- **kernel_size** (int): The kernel size of the convolutional layers. Default is 5.
- **cond_predict_scale** (bool): Whether to use FiLM layers to condition the model on the observation sequence. Default is True.
- **obs_as_global_cond** (bool): If True, flatten the observation tensor and use it as the global condition. Otherwise, use convolutional blocks to process the sequential observation tensor. Default is True.
- **dim_mult** (List[int]): The multiplier for the number of channels in each layer. Default is [1, 2, 2, 2].
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(..., Ta, act_dim)`.
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(..., 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(..., To, obs_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., Ta, act_dim)`.
