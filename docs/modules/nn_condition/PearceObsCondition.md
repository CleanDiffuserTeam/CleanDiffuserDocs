---
layout: default
title: PearceObsCondition
nav_order: 3.3.5
parent: nn_condition
grand_parent: MODULES
---

# **PearceObsCondition**

> **CLASS** cleandiffuser.nn_diffusion.PearceObsCondition(obs_dim: int, emb_dim: int, flatten: bool = False, dropout: float = 0.25) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/pearce_obs_condition.py)

A MLP condition processor proposed in [Diffusion Behavior Clone (DBC)](https://arxiv.org/abs/2301.10677). The raw context tensor is a sequence of low-dim observations. It encodes each observation frame using the same MLP, then flattens them to create a condition embedding.

**Parameters:**
- **obs_dim** (int): The dimension of the observation. Suppose the observation has shape `(b, To, obs_dim)`, where `b` is the batch size, `To` is the number of frames, and `obs_dim` is the dimension of each frame.
- **emb_dim** (int): The dimension of the condition embedding. Default is 128.
- **flatten** (bool): Whether to flatten the condition embedding. Default is False.
- **dropout** (float): The label dropout rate. Default is 0.25.

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(b, To, obs_dim)`.
- **mask** (Optional[torch.Tensor]): The mask tensor. Default is None. None means no mask.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, To * emb_dim)` if `flatten` is True, otherwise `(b, To, emb_dim)`. Each element in the batch has a probability of `dropout` to be zeros.
