---
layout: default
title: LinearCondition
nav_order: 3.3.3
parent: nn_condition
grand_parent: MODULES
---

# **LinearCondition**

> **CLASS** cleandiffuser.nn_diffusion.LinearCondition(in_dim: int, out_dim: int, dropout: float = 0.25) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/mlp.py)

A linear condition processor. It does an affine transformation on the raw context tensor.

**Parameters:**
- **in_dim** (int): The dimension of the input tensor $$\bar{\bm c}$$.
- **out_dim** (int): The dimension of the output tensor $$\bm c$$.
- **dropout** (float): The label dropout rate for the context tensor. Default is 0.25.

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(b, ..., in_dim)`.
- **mask** (Optional[torch.Tensor]): The mask tensor. Default is None. None means no mask.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, ..., out_dim)`. Each element in the batch has a probability of `dropout` to be zeros.
