---
layout: default
title: DQLMlp
nav_order: 3.2.4
parent: nn_diffusion
grand_parent: MODULES
---

# **DQLMlp**

> **CLASS** cleandiffuser.nn_diffusion.DQLMlp(obs_dim: int, act_dim: int, emb_dim: int = 16, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/dqlmlp.py)

A simple MLP neural network backbone for diffusion model, proposed in [Diffusion Q-Learning (DQL)](https://arxiv.org/abs/2208.06193).
It takes the current observation as the context tensor and generates the action to execute.

**Parameters:**
- **obs_dim** (int): The dimension of the observation tensor $$\bm o_t$$.
- **act_dim** (int): The dimension of the action tensor $$\bm x_t$$.
- **emb_dim** (int): The dimension of the time embedding. Default is 16.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(..., act_dim)`
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(..., 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(..., obs_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., act_dim)`.
