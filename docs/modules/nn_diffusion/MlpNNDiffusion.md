---
layout: default
title: MlpNNDiffusion
nav_order: 3.2.2
parent: nn_diffusion
grand_parent: MODULES
---

# **MlpNNDiffusion**

> **CLASS** cleandiffuser.nn_diffusion.MlpNNDiffusion(x_dim: int, emb_dim: int, hidden_dims: List[int] = (256, 256), activation: torch.nn.Module = torch.nn.ReLU(), timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/mlps.py)

A simple MLP neural network backbone for diffusion models. It directly concatenates the input tensor $$\bm x_t$$ with the embedding tensor (time embedding plus context embedding) and passes it through a MLP to get the output tensor.

**Parameters:**
- **x_dim** (int): The dimension of the input tensor $$\bm x_t$$.
- **emb_dim** (int): The dimension of the time embedding.
- **hidden_dims** (List[int]): The dimensions of the hidden layers of the MLP. Default is [256, 256].
- **activation** (torch.nn.Module): The activation function of the hidden layers. Default is `torch.nn.ReLU()`.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(..., x_dim)`
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(..., 1)`.
- **c** (Optional[torch.Tensor]): The context tensor $$\bm c$$ in shape `(..., emb_dim)`. Default is None.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(..., x_dim)`.
