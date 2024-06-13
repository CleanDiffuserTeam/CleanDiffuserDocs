---
layout: default
title: Getting Started
nav_order: 1.3
parent: INTRODUCTION
---

# Getting Started

The `cleandiffuser` folder contains the core components of the CleanDiffuser codebase, including `Diffusion Models`, `Network Architectures`, and `Guided Sampling`. It also provides unified `Env and Dataset Interfaces`.

In the `tutorials` folder, we provide the simplest runnable tutorials and algorithms, which can be understood in conjunction with the documentation.

In CleanDiffuser, we can combine independent modules to algorithms pipelines like building blocks. In the `pipelines` folder, we provide all the algorithms currently implemented in CleanDiffuser. By linking with the Hydra configurations in the `configs` folder, you can reproduce the results presented in the papers:

You can simply run each algorithm with the default environment and configuration without any additional setup, for example:

```bash
# DiffusionPolicy with Chi_UNet in lift-ph
python pipelines/dp_pusht.py
# Diffuser in halfcheetah-medium-expert-v2
python pipelines/diffuser_d4rl_mujoco.py
```

Thanks to Hydra, CleanDiffuser also supports flexible running of algorithms through CLI or directly modifying the corresponding configuration files. We provide some examples:

```bash
# Load PushT config
python pipelines/dp_pusht.py --config-path=../configs/dp/pusht/dit --config-name=pusht
# Load PushT config and overwrite some hyperparameters
python pipelines/dp_pusht.py --config-path=../configs/dp/pusht/dit --config-name=pusht dataset_path=path/to/dataset seed=42 device=cuda:1
# Train Diffuser in hopper-medium-v2 task
python pipelines/diffuser_d4rl_mujoco.py task=hopper-medium-v2 
```

In CleanDiffuser, we provide a mode option to switch between **training** `(mode=train)` or **inference** `(mode=inference)` of the model:

```bash
# Imitation learning environment
python pipelines/dp_pusht.py mode=inference model_path=path/to/checkpoint
# Reinforcement learning environment
python pipelines/diffuser_d4rl_mujoco.py mode=inference ckpt=latest
```