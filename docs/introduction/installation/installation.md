---
layout: default
title: Installation
nav_order: 1.2
parent: INTRODUCTION
---

# Installation

We recommend installing and experiencing CleanDiffuser through a Conda virtual environment.

First, install the dependencies related to the mujoco-py environment. For more details, see https://github.com/openai/mujoco-py#install-mujoco

```bash
apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Download CleanDiffuser and add this folder to your PYTHONPATH. You can also add it to .bashrc for convenience:
```bash
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
export PYTHONPATH=$PYTHONPATH:/path/to/CleanDiffuser
```

Install the Conda virtual environment and PyTorch:
```bash
conda create -n cleandiffuser python==3.9
conda activate cleandiffuser
# pytorch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
```

Install the remaining dependencies:
```bash
pip install -r requirements.txt

# If you need to run D4RL-related environments, install D4RL additionally:
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

If you need to reproduce imitation learning related environments (PushT, Kitchen, Robomimic), you need to download the datasets additionally. We recommend downloading the corresponding compressed files from [Datasets](https://diffusion-policy.cs.columbia.edu/data/training/). We provide the default dataset path as `dev/`:

```bash
dev/
.
├── kitchen
├── pusht_cchi_v7_replay.zarr
├── robomimic
```