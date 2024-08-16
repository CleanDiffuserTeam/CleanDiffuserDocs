---
layout: default
title: PearceTransformer
nav_order: 3.2.9
parent: nn_diffusion
grand_parent: MODULES
---

# **PearceTransformer**

> **CLASS** cleandiffuser.nn_diffusion.PearceTransformer(act_dim: int, To: int = 1, emb_dim: int = 128, trans_emb_dim: int = 64, nhead: int = 16, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/pearcetransformer.py)

A stacked transformer encoder layers for diffusion models. It is proposed in [Diffusion Behavior Clone (DBC)](https://arxiv.org/abs/2301.10677). It assumes the input tensor $$\bm x_t$$ is the action tensor and the context tensor $$\bm c$$ is the observation sequence tensor (contains the current observation and the previous observations).

**Parameters:**
- **act_dim** (int): The dimension of the action tensor $$\bm x_t$$.
- **To** (int): The number of observations to consider. 1 means only the current observation. Default is 1.
- **emb_dim** (int): The dimension of the time embedding. Default is 128.
- **trans_emb_dim** (int): The dimension of the each head in the transformer encoder. Default is 64.
- **nhead** (int): The number of heads in the multiheadattention models. Default is 16.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(b, act_dim)`.
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(b, 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(b, To, emb_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, act_dim)`.
