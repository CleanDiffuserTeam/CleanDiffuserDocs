---
layout: default
title: Special. Consistency Policy
nav_order: 2.5
parent: TUTORIALS
---

# Special. Consistency Policy

[TOC]

**[2024-06-24] 🥰 We have added Consistency Models into CleanDiffuser. With one model, you can do either Consistency Distillation or Consistency Training! (Note: Our consistency training implementation uses the improved version, see https://arxiv.org/abs/2310.14189, which will provide you with maximum performance support.) **

In this tutorial, we will together explore using the Consistency Model as the policy in IDQL, experiencing the thrill of one-step generation speed.

## 1. How to use the Consistency Model?

The Consistency Model defines a consistency function.  A consistency function has the property of self-consistency: its outputs are consistent for arbitrary pairs of (x_t, t) that belong to the same PF ODE trajectory. The Consistency Model needs to be distilled from a pre-trained EDM or learned directly through consistency training loss to learn such a consistency function. This self-consistency property allows the Consistency Model in theory to achieve one-step generation. 

If we want to use Consistency Distillation, we need to 1. pre-train an EDM, 2. train the Consistency Model with consistency distillation loss. If we want to use Consistency Training, we can directly train the Consistency Model with consistency training loss. In CleanDiffuser, both methods can be easily accomplished. We need only:

```python
from cleandiffuser.diffusion import ContinuousConsistencyModel

# Consistency Training
actor = ContinuousConsistencyModel(nn_diffusion, nn_condition)
actor.update(act, obs, loss_type="training")

# Consistency Distillation
actor = ContinuousConsistencyModel(nn_diffusion, nn_condition)
actor.prepare_distillation(pretrained_edm_actor)
actor.update(act, obs, loss_type="distillation")
```

Except for Consistency Distillation, which requires an additional initialization line of code, the usage of Consistency Models in CleanDiffuser is identical to other Diffusion Models. You only need to make extremely minor adjustments to use the Consistency Model backbone in your algorithm.

## 2. Let's try a "Consistency IDQL"!

Let's briefly review IDQL. IDQL is an Offline RL algorithm that independently learns IQL value functions and a behavior clone diffusion policy. During Inference, it uses IQL's value estimation to reselect the action to be executed from many diffusion-generated actions. Therefore, to test Consistency Distillation, we need to follow the process: `iql_training -> edm_training -> cd_training -> inference`. To test Consistency Training, we need to follow the process: `iql_training -> ct_training -> inference`.

### 2.1 Prepare Environments, Datasets, and IQL

A simple IQL value training code that you don't need to pay too much attention to.

```python
import os
from copy import deepcopy

import d4rl
import gym
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousEDM
from cleandiffuser.diffusion.consistency_model import ContinuousConsistencyModel
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import (IDQLQNet, IDQLVNet)


seed = 0
device = "cuda:0"
env_name = "halfcheetah-medium-v2"
weight_temperature = 100. # 10 for me / 100 for m / 400 for mr
mode = "iql_training"

set_seed(seed)
save_path = f'tutorials/results/sp_consistency_policy/{env_name}/'
if os.path.exists(save_path) is False:
    os.makedirs(save_path)

# ---------------------- Create Dataset ----------------------
env = gym.make(env_name)
dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), True)
dataloader = DataLoader(
    dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
obs_dim, act_dim = dataset.o_dim, dataset.a_dim

""" MODE1: IQL Training
    
In IDQL, the Diffusion model simply behavior clones the dataset 
and reselects actions during inference through IQL's value estimation. 
Therefore, we need to train an IQL here.
"""
if mode == "iql_training":

    # Create IQL Networks
    iql_q = IDQLQNet(obs_dim, act_dim).to(device)
    iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
    iql_v = IDQLVNet(obs_dim).to(device)

    q_optim = torch.optim.Adam(iql_q.parameters(), lr=3e-4)
    v_optim = torch.optim.Adam(iql_v.parameters(), lr=3e-4)

    q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=1_000_000)
    v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=1_000_000)

    iql_q.train()
    iql_v.train()

    # Begin Training
    n_gradient_step = 0
    log = {"q_loss": 0., "v_loss": 0.}
    for batch in loop_dataloader(dataloader):

        obs, next_obs = batch["obs"]["state"].to(device), batch["next_obs"]["state"].to(device)
        act = batch["act"].to(device)
        rew = batch["rew"].to(device)
        tml = batch["tml"].to(device)

        q = iql_q_target(obs, act)
        v = iql_v(obs)
        v_loss = (torch.abs(0.7 - ((q - v) < 0).float()) * (q - v) ** 2).mean()

        v_optim.zero_grad()
        v_loss.backward()
        v_optim.step()

        with torch.no_grad():
            td_target = rew + 0.99 * (1 - tml) * iql_v(next_obs)
        q1, q2 = iql_q.both(obs, act)
        q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
        q_optim.zero_grad()
        q_loss.backward()
        q_optim.step()

        q_lr_scheduler.step()
        v_lr_scheduler.step()

        for param, target_param in zip(iql_q.parameters(), iql_q_target.parameters()):
            target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

        # Logging
        log["q_loss"] += q_loss.item()
        log["v_loss"] += v_loss.item()

        if (n_gradient_step + 1) % 1000 == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["q_loss"] /= 1000
            log["v_loss"] /= 1000
            print(log)
            log = {"q_loss": 0., "v_loss": 0.}

        # Saving
        if (n_gradient_step + 1) % 200_000 == 0:
            torch.save({
                "iql_q": iql_q.state_dict(),
                "iql_q_target": iql_q_target.state_dict(),
                "iql_v": iql_v.state_dict(),
            }, save_path + f"iql_ckpt_{n_gradient_step + 1}.pt")
            torch.save({
                "iql_q": iql_q.state_dict(),
                "iql_q_target": iql_q_target.state_dict(),
                "iql_v": iql_v.state_dict(),
            }, save_path + f"iql_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= 1_000_000:
            break
```

### 2.2 EDM Training & Consistency Distillation (CD)

In the first part of the code, we will train an EDM behavior clone actor with parameter settings following the default in IDQL `pipelines`, serving as the backbone for distillation. In the second part, we will perform consistency distillation for 200k gradient steps.

```python
elif mode == "edm_training":
        
    """ MODE2: EDM Training

    Consistency Distillation (CD) requires a well-trained EDM backbone. 
    If you only want to test Consistency Training, this step is not necessary.
    """

    nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
    nn_condition = IdentityCondition(dropout=0.0)

    actor = ContinuousEDM(
        nn_diffusion, nn_condition, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        ema_rate=0.9999, device=device)

    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=1_000_000)

    actor.train()

    n_gradient_step = 0
    log = {"bc_loss": 0.}

    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)

        bc_loss = actor.update(act, obs)["loss"]
        actor_lr_scheduler.step()

        # Logging
        log["bc_loss"] += bc_loss

        if (n_gradient_step + 1) % 1000 == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["bc_loss"] /= 1000
            print(log)
            log = {"bc_loss": 0.}

        # Saving
        if (n_gradient_step + 1) % 200_000 == 0:
            actor.save(save_path + f"edm_ckpt_{n_gradient_step + 1}.pt")
            actor.save(save_path + f"edm_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= 1_000_000:
            break

elif mode == "cd_training":

    """ MODE3: Consistency Distillation

    Train the Consistency Model with a pre-trained EDM.
    """

    # Load pre-trained EDM
    nn_diffusion_edm = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
    nn_condition_edm = IdentityCondition(dropout=0.0)

    edm_actor = ContinuousEDM(
        nn_diffusion_edm, nn_condition_edm, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        ema_rate=0.9999, device=device)

    edm_actor.load(save_path + f"edm_ckpt_latest.pt")
    edm_actor.eval()

    # Create Consistency Model
    nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
    nn_condition = IdentityCondition(dropout=0.0)

    actor = ContinuousConsistencyModel(
        nn_diffusion, nn_condition, optim_params={"lr": 3e-4}, 
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        ema_rate=0.9999, device=device)

    actor.prepare_distillation(edm_actor, distillation_N=18)

    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=200_000)

    actor.train()

    n_gradient_step = 0
    log = {"loss": 0.}
    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)

        loss = actor.update(act, obs, loss_type="distillation")["loss"]

        log["loss"] += loss

        if (n_gradient_step + 1) % 1000 == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["loss"] /= 1000
            print(log)
            log = {"loss": 0.}

        if (n_gradient_step + 1) % 200_000 == 0:
            actor.save(save_path + f"cd_ckpt_{n_gradient_step + 1}.pt")
            actor.save(save_path + f"cd_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= 200_000:
            break
```

### 2.3 Consistency Training (CT)

The following code directly performs Consistency Training, training a Consistency model without relying on other pre-trained models.

```python
elif mode == "ct_training":

    """ MODE4: Consistency Training

    Train the Consistency Model without relying on any pre-trained Models.
    """

    # As suggested in "IMPROVED TECHNIQUES FOR TRAINING CONSISTENCY MODELS", the Fourier scale is set to 0.02 instead of default 16.0.
    nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier", timestep_emb_params={"scale": 0.02})
    nn_condition = IdentityCondition(dropout=0.0)

    actor = ContinuousConsistencyModel(
        nn_diffusion, nn_condition, optim_params={"lr": 3e-4},
        curriculum_cycle=1000000,
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        ema_rate=0.9999, device=device)

    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=1_000_000)

    actor.train()

    n_gradient_step = 0
    log = {"bc_loss": 0., "unweighted_bc_loss": 0.}

    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"].to(device)
        act = batch["act"].to(device)

        # -- Policy Training
        _log = actor.update(act, obs)

        actor_lr_scheduler.step()

        # ----------- Logging ------------
        log["bc_loss"] += _log["loss"]
        log["unweighted_bc_loss"] += _log["unweighted_loss"]

        if (n_gradient_step + 1) % 1000 == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["bc_loss"] /= 1000
            log["unweighted_bc_loss"] /= 1000
            log["curriculum_process"] = actor.cur_logger.curriculum_process
            log["Nk"] = actor.cur_logger.Nk
            print(log)
            log = {"bc_loss": 0., "unweighted_bc_loss": 0.}

        # ----------- Saving ------------
        if (n_gradient_step + 1) % 200_000 == 0:
            actor.save(save_path + f"ct_ckpt_{n_gradient_step + 1}.pt")
            actor.save(save_path + f"ct_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= 1_000_000:
            break
```

### 2.4 Inference Comparisons

We test the performance of the models on HalfCheetah task in D4RL-MuJoCo using the medium-expert, medium, and medium-replay datasets. The compared algorithms include training an EDM for 1M steps, CD for 200K steps, and CT for 1M steps. The results are as follows:

| Environment    | Sampling Steps (NFE) | EDM (1M) | CD (200K) | CT (1M)  |
| -------------- | :------------------- | -------- | --------- | -------- |
| HalfCheetah-me | 1                    | 44.1±1.1 | 46.0±2.5  | 44.6±1.4 |
|                | 2                    | 93.2±0.1 | 73.3±2.7  | 38.6±2.9 |
|                | 4                    | 92.8±0.1 | 58.6±0.2  | 37.7±1.0 |
| HalfCheetah-m  | 1                    | 49.9±0.2 | 50.6±0.1  | 53.2±0.1 |
|                | 2                    | 44.7±0.1 | 52.2±0.1  | 54.0±0.0 |
|                | 4                    | 49.7±0.0 | 53.3±0.1  | 54.1±0.0 |
| HalfCheetah-mr | 1                    | 43.8±0.2 | 42.2±0.4  | 46.7±0.3 |
|                | 2                    | 42.8±0.1 | 47.8±0.6  | 48.5±0.1 |
|                | 4                    | 46.3±0.1 | 48.3±0.3  | 48.7±0.3 |

Our reported results correspond to the mean and standard error over 150 episode seeds. It can be observed that the Consistency Model indeed shows good performance with very few sampling steps. HalfCheetah-medium-expert is an exception, where both CD and CT perform poorly. This could be due to hyperparameters; we did not perform any hyperparameter tuning for these algorithms. The parameters for IDQL are default parameters from `pipelines`, and the parameters for the Consistency model are default parameters from its original paper (for image generation), which may not be a perfect match for decision-making tasks.

## 3. Summary

In this tutorial, we replace the diffusion backbone of the policy in IDQL with the Consistency Model (which includes both CD and CT). In CleanDiffuser, the training and inference code for the Consistency Model is almost identical to other Diffusion Models. You only need to make minimal changes to your code to deploy and use Consistency Models. If you are still struggling with the slow decision-making speed of your diffusion decision-making algorithm, you should try this one! Consistency Models are all you need.
