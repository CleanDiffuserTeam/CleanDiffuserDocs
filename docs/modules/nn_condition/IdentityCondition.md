---
layout: default
title: IdentityCondition
nav_order: 3.3.2
parent: nn_condition
grand_parent: MODULES
---

# **IdentityCondition**

> **CLASS** cleandiffuser.nn_diffusion.IdentityCondition(dropout: float = 0.25) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/base_nn_condition.py)

An identity condition processor. It simply assumes $$\bar{\bm c}=\bm c$$. Therefore, the dummy tensor $$\bm\Phi$$ is zeros in the raw context tensor space. Be sure that the zeros tensor has no meaning in the raw context tensor space.

**Parameters:**
- **dropout** (float): The label dropout rate for the context tensor. Default is 0.25.

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(..., *c_shape)`.
- **mask** (Optional[torch.Tensor]): The mask tensor. Default is None. None means no mask.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., *c_shape)`. Each element in the batch has a probability of `dropout` to be zeros.
