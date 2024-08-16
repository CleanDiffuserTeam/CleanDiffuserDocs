---
layout: default
title: SfBCUNet
nav_order: 3.2.8
parent: nn_diffusion
grand_parent: MODULES
---

# **SfBCUNet**

> **CLASS** cleandiffuser.nn_diffusion.SfBCUNet(act_dim: int, emb_dim: int = 64, hidden_dims: List[int] = [512, 256, 128], timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/chiunet.py)

A modified U-Net architecture to process low-dim vectors, proposed in [SfBC](https://openreview.net/forum?id=42zs3qa2kpy). Compared to the original U-Net, it uses MLPs instead of convolutional layers.

**Parameters:**
- **act_dim** (int): The dimension of the action tensor $$\bm x_t$$.
- **emb_dim** (int): The dimension of the time embedding. Default is 256.
- **hidden_dims** (List[int]): The dimensions of the hidden layers of the MLP. Default is [512, 256, 128].
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(..., act_dim)`.
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(..., 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(..., emb_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., act_dim)`.
