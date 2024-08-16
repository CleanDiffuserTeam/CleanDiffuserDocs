---
layout: default
title: BaseNNDiffusion
nav_order: 3.2.1
parent: nn_diffusion
grand_parent: MODULES
---

# **BaseNNDiffusion**

> **CLASS** cleandiffuser.nn_diffusion.BaseNNDiffusion(emb_dim: int, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/base_nn_diffusion.py)

The base class for neural network (NN) backbones for diffusion models. It can be the scaled score function in VE/VP-SDEs, the drift force in ODEs, or any other NN parameterized function in the diffusion models. Suppose the generated data $$\bm x\in\mathbb R^N$$, the NN backbone can be represented as $$\bm f_{\theta}(\bm x_t,t,\bm c)\in\mathbb R^N$$, where $$\bm x_t$$ is the noisy data at denoising time $$t$$ and $$\bm c$$ is the context information (can be a dummy variable, e.g., `None`, when not used). In practice, since $$t$$ is a scalar, it is a common choice to use a positional encoding (for discrete $$t$$) or a fourier encoding (for continuous $$t$$) to pre-encode the time information to a fixed-length vector. So we define the base class as follows:

**Parameters:**
- **emb_dim** (int): The dimension of the time embedding.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.


As mentioned above, we expect the derived classes to output a tensor with the same shape as the input tensor $$\bm x_t$$. The derived classes should implement the following `forward` method:

> forward(x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(..., *x_shape)`
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(..., 1)`.
- **c** (Optional[torch.Tensor]): The context tensor $$\bm c$$ in shape `(..., *c_shape)`. Default is None.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., *x_shape)`.

> **Note:** This class is an abstract class and should not be instantiated directly.