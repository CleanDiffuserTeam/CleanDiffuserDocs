---
layout: default
title: MLPCondition
nav_order: 3.3.4
parent: nn_condition
grand_parent: MODULES
---

# **MLPCondition**

> **CLASS** cleandiffuser.nn_diffusion.MLPCondition(in_dim: int, out_dim: int, hidden_dims: List[int], act: nn.Module = nn.LeakyReLU(), dropout: float = 0.25) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/mlp.py)

A multi-layer perceptron (MLP) condition processor. The raw context tensor $$\bar{\bm c}$$ is passed through a MLP to get the output tensor $$\bm c$$.

**Parameters:**
- **in_dim** (int): The dimension of the input tensor $$\bar{\bm c}$$.
- **out_dim** (int): The dimension of the output tensor $$\bm c$$.
- **hidden_dims** (List[int]): The dimensions of the hidden layers of the MLP.
- **act** (nn.Module): The activation function of the hidden layers. Default is `nn.LeakyReLU()`.
- **dropout** (float): The label dropout rate for the context tensor. Default is 0.25.

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(b, ..., in_dim)`.
- **mask** (Optional[torch.Tensor]): The mask tensor. Default is None. None means no mask.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, ..., out_dim)`. Each element in the batch has a probability of `dropout` to be zeros.
