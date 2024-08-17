---
layout: default
title: ChiTransformer
nav_order: 3.2.11
parent: nn_diffusion
grand_parent: MODULES
---

# **ChiTransformer**

> **CLASS** cleandiffuser.nn_diffusion.ChiTransformer(act_dim: int, obs_dim: int, Ta: int, To: int, d_model: int = 256, nhead: int = 4, num_layers: int = 8, p_drop_emb: float = 0.0, p_drop_attn: float = 0.3, n_cond_layers: int = 0, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/chitransformer.py)

A modified Transformer for 1D input tensor $$\bm x_t$$, proposed in [DiffusionPolicy](https://diffusion-policy.cs.columbia.edu/). Compared to the original Transformer, it incorporates a carefully designed masking mechanism to handle decision-making trajectories. It generates action trajectory sequences.

**Parameters:**
- **act_dim** (int): The dimension of the action tensor $$\bm x_t$$.
- **obs_dim** (int): The dimension of the observation tensor $$\bm c$$.
- **Ta** (int): The number of actions to predict.
- **To** (int): The number of observations to consider. 1 means only the current observation.
- **d_model** (int): The dimension of the transformer model. Default is 256.
- **nhead** (int): The number of heads in the multi-head attention. Default is 4.
- **num_layers** (int): The number of transformer layers. Default is 8.
- **p_drop_emb** (float): The dropout rate for the embedding layer. Default is 0.0.
- **p_drop_attn** (float): The dropout rate for the attention layer. Default is 0.3.
- **n_cond_layers** (int): The number of layers to condition on the observation sequence. Default is 0.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(b, Ta, act_dim)`.
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(b, 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(b, To, obs_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, Ta, in_dim)`.
