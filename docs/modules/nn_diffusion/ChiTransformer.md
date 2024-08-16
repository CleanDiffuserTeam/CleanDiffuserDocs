---
layout: default
title: ChiTransformer
nav_order: 3.2.11
parent: nn_diffusion
grand_parent: MODULES
---

# **ChiTransformer**

> **CLASS** cleandiffuser.nn_diffusion.ChiTransformer(act_dim: int, obs_dim: int, Ta: int, To: int, d_model: int = 256,
    
    
    in_dim: int, emb_dim: int, d_model: int = 384, n_heads: int = 6, depth: int = 12, dropout: float = 0.0, timestep_emb_type: str = "positional", timestep_emb_params: Optional[dict] = None) [[SOURCE]](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/nn_diffusion/dit.py)

A modified Diffusion Transformer for 1D input tensor $$\bm x_t$$, proposed in [AlignDiff](https://arxiv.org/abs/2310.02054). It generates trajectory sequences. One important feature is that it can handle variable-length input sequences, i.e., the sequence length during training and inference can be different.

**Parameters:**
- **in_dim** (int): The dimension of the input tensor $$\bm x_t$$.
- **emb_dim** (int): The dimension of the time embedding.
- **d_model** (int): The dimension of the transformer model. Default is 384.
- **n_heads** (int): The number of heads in the multi-head attention. Default is 6.
- **depth** (int): The number of DiT blocks. Default is 12.
- **dropout** (float): The dropout rate. Default is 0.0.
- **timestep_emb_type** (str): The type of the time embedding. It can be either "positional" or "fourier". Default is "positional".
- **timestep_emb_params** (Optional[dict]): The parameters for the time embedding. Default is None.

> forward(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor

**Parameters:**
- **x** (torch.Tensor): The input tensor $$\bm x_t$$ in shape `(b, seq_len, in_dim)`.
- **t** (torch.Tensor): The time tensor $$t$$ in shape `(b, 1)`.
- **c** (torch.Tensor): The context tensor $$\bm c$$ in shape `(b, emb_dim)`.

**Returns:**
- **torch.Tensor**: The output tensor in shape `(b, seq_len, in_dim)`.
