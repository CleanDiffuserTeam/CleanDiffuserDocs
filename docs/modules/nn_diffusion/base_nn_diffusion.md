---
layout: default
title: base_nn_diffusion
nav_order: 3.2.1
parent: nn_diffusion
grand_parent: MODULES
---

# **base_nn_diffusion**

> **CLASS** cleandiffuser.nn_diffusion.BaseNNDiffusion(emb_dim: int, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/base_nn_diffusion.py)

The base class for neural network backbones for diffusion models. It can be the scaled score function in VE/VP-SDEs, the drift force in ODEs, or any other neural network parameterized function in the diffusion models. Suppose the generated data 
$$
\bm x\in\mathbb R^N
$$