---
layout: default
title: BaseNNCondition
nav_order: 3.3.1
parent: nn_condition
grand_parent: MODULES
---

# **BaseNNCondition**

> **CLASS** cleandiffuser.nn_diffusion.BaseNNCondition() [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_condition/base_nn_condition.py)

The base class for condition process. Recall that the NN backbone for diffusion model can be represented as $$\bm f_{\theta}(\bm x_t,t,\bm c)$$, which $$\bm c$$ is a tensor of context information. However, in practice, the context information can be very complex, including multi-modal data, high-dimensional data, etc. To simplify the context information, we can use a separate neural network $$\bm\zeta_\phi(\bar{\bm c})=\bm c$$ to process it, where $$\bar{\bm c}$$ is the complex raw context information and $$\bm c$$ is the simplified context tensor.

When we need a classifier-free guidance (CFG), our diffusion NN backbone must recognize a dummy context tensor $$\bm\Phi$$ such that $$\bm f_{\theta}(\bm x_t,t,\bm\Phi)=\bm f_{\theta}(\bm x_t,t)$$ means unconditional diffusion. So it is a common choice to define the dummy tensor as zeros, i.e., $$\bm\Phi=\bm 0$$. And randomly drop out some elements of the context tensor when training the NN backbone. To achieve this, we define the base class as follows:

**Parameters:**
- **None**

> forward(condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **condition** (torch.Tensor): The context tensor in shape `(..., *c_shape)`.
- **mask** (Optional[torch.Tensor]): The mask tensor in shape `(..., *c_shape)`. Default is None.

**Returns:**
- **None**

