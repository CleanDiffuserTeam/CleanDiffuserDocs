---
layout: default
title: Installation
nav_order: 1.2
parent: INTRODUCTION
---

# Installation

### 1. Create and activate conda environment
```bash
$ conda create -n cleandiffuser python==3.9
$ conda activate cleandiffuser
```

### 2. Install PyTorch
Install `torch>1.0.0,<2.3.0` that is compatible with your CUDA version. For example, `PyTorch 2.2.2` with `CUDA 12.1`:
```bash
$ conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install CleanDiffuser from source
```bash
$ git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
$ cd CleanDiffuser
$ pip install -e .
```

### 4. Additional installations (Optional)
For users who need to run `pipelines` and reproduce the results of the paper, they will need to install RL simulators.

First, install the dependencies related to the mujoco-py environment. For more details, see https://github.com/openai/mujoco-py#install-mujoco

```bash
$ sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
```
```bash
# Install D4RL from source (recommended)
$ cd <PATH_TO_D4RL_INSTALL_DIR>
$ git clone https://github.com/Farama-Foundation/D4RL.git
$ cd D4RL
$ pip install -e .
# Install Robomimic from source (recommended)
$ cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>
$ git clone https://github.com/ARISE-Initiative/robomimic.git
$ cd robomimic
$ pip install -e .
$ cd <PATH_TO_ROBOSUITE_INSTALL_DIR>
$ git clone https://github.com/ARISE-Initiative/robosuite.git
$ cd robosuite
$ pip install -e .
```

> **Note:** The latest version of dependencies running the `robomimic image` still has compatibility issues, and we are actively working on a fix. The temporary solution is to downgrade the `gym` version to `0.21.0`: pip install setuptools==65.5.0 pip==21, pip install gym==0.21.0

Try it now!   
```bash
# Tutorial
$ python tutorials/1_a_minimal_DBC_implementation.py
# Reinforcement Learning
$ python pipelines/diffuser_d4rl_mujoco.py
# Imitation Learning (need to download the dataset, see below)
$ python pipelines/dp_pusht.py
```
If you need to reproduce Imitation Learning environments (`pusht`, `kitchen`, `robomimic`), you need to download the datasets additionally. We recommend downloading the corresponding compressed files from [Datasets](https://diffusion-policy.cs.columbia.edu/data/training/). We provide the default dataset path as `dev/`:
```bash
dev/
.
├── kitchen
├── pusht
├── robomimic
```
