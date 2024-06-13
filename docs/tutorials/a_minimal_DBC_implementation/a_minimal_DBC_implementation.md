---
layout: default
title: 1 A Minimal DBC Implementation
nav_order: 2.1
parent: TUTORIALS
---

# A Minimal DBC Implementation

In this tutorial, we'll explore how to implement a basic Diffusion Behavior Clone (DBC) using CleanDiffuser. DBC is an imitation learning algorithm that aims to replicate behaviors from an offline dataset. It leverages a diffusion model to generate samples from the policy distribution `\pi_\theta(a|s)`.

## 1 Setting Up the Environment

Since the imitation learning algorithms simply imitate the behaviors in the offline dataset, we cannot expect DBC to perform well on low-quality datasets. In this tutorial, we use `kitchen-complete-v0` from D4RL as our test environment. In this environment, our agent needs to complete various kitchen tasks within a limited time, such as moving the kettle, opening the microwave, and so on. The `complete` dataset provides many demonstrations that can fully complete the tasks.

```python
import d4rl
import gym

# --------------- Setting Up the Environment ---------------
env = gym.make("kitchen-complete-v0")
dataset = d4rl.qlearning_dataset(env)
obs_dim, act_dim = dataset['observations'].shape[-1], dataset['actions'].shape[-1]
size = len(dataset['observations'])
```

## 2 Choosing the Network Architecture

If we consider diffusion models as the "soul", then neural networks serve as its "shell". The probability distribution we aim to fit is an action distribution conditioned on state. To begin, we require a `nn_condition` for preprocessing the condition, i.e., states. We utilize the `PearceObsCondition` implemented in CleanDiffuser to encode the input observation sequence (a single observation if historical observations are not used as context) into features of shape `(batch_size, To*emb_dim)`, which is then fed into `nn_diffusion`. The `nn_diffusion` is employed to estimate the unknown terms in the diffusion reverse process. In this tutorial, we utilize `PearceMlp` implemented in CleanDiffuser to predict the scaled score function. In the implementation, we only need to import these two classes and initialize them, both of which inherit from PyTorch's `nn.Module`.

```python
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp

# --------------- Network Architecture -----------------
nn_diffusion = PearceMlp(act_dim, To=1, emb_dim=64, hidden_dim=256, timestep_emb_type="positional")
# (bs, act_dim), (bs, ), (bs, To * emb_dim) -> (bs, act_dim)
nn_condition = PearceObsCondition(obs_dim, emb_dim=64, flatten=True, dropout=0.0)
# (bs, To, obs_dim) -> (bs, To * emb_dim)
```

## 3 Crafting the Diffusion Actor

We choose `DiscreteDiffusionSDE`, which optimizes a score-matching loss to learn the score function in VPSDE and discretizes the time interval of the diffusion process into a finite number of timesteps. During sampling, we can choose any number of sampling steps greater than 1 and not less than `diffusion_steps`, and we can select a range of available solvers. When instantiating this class, we also define several other parameters. Setting `predict_noise=False` will instruct the NN to directly predict denoised actions rather than noise. `optim_params` will override the default creation parameters of the optimizer. `x_max` and `x_min` will clip the range of generated data during the sampling process to reduce out-of-distribution sampling.

```python
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE

# --------------- Diffusion Model Actor --------------------
actor = DiscreteDiffusionSDE(
    nn_diffusion, nn_condition, predict_noise=False, optim_params={"lr": 3e-4},
    x_max=+1. * torch.ones((1, act_dim)),
    x_min=-1. * torch.ones((1, act_dim)),
    diffusion_steps=5, ema_rate=0.9999, device=device)
```

## 4 Training and Evaluation

Generally, training the diffusion model only requires using the `update` method, providing the raw data and the corresponding generation conditions for a single gradient update (the condition parameter is not required when not needed). During sampling, we simply need to call the `sample` method to obtain the generated data from the diffusion model and some accompanying logs produced during this process.

```python
# --------------- Training -------------------
actor.train()

avg_loss = 0.

# train for 100,000 gradient steps (Actually it is not enough for a well-trained model.)
for t in range(100000):

    # sample a batch
    idx = np.random.randint(0, size, (256,))
    obs = torch.tensor(dataset['observations'][idx], device=device).float()
    act = torch.tensor(dataset['actions'][idx], device=device).float()

    # one-step update
    avg_loss += actor.update(act, obs)["loss"]

    # logging
    if (t + 1) % 1000 == 0:
        print(f'[t={t + 1}] {avg_loss / 1000}')
        avg_loss = 0.

# model saving
savepath = "tutorials/results/1_Minimal_DBC/"
if not os.path.exists(savepath):
    os.makedirs(savepath)
actor.save(savepath + "diffusion.pt")
            
# -------------- Inference -----------------
savepath = "tutorials/results/1_Minimal_DBC/"
actor.load(savepath + "diffusion.pt")
actor.eval()

# concurrently evaluate 50 environments
env_eval = gym.vector.make("kitchen-complete-v0", num_envs=50)

obs, cum_done, cum_rew = env_eval.reset(), 0., 0.

"""
`prior` is used to provide prior information, such as known parts of the generated data. 
In this context, it is evident that we do not have any prior information to provide.
"""
prior = torch.zeros((50, act_dim), device=device)
for t in range(280):

    # sample with DDPM and a sampling step of 5
    act, log = actor.sample(
        prior, solver="ddpm", n_samples=50, sample_steps=5,
        temperature=0.5, w_cfg=1.0,
        condition_cfg=torch.tensor(obs, device=device, dtype=torch.float32))
    act = act.cpu().numpy()

    obs, rew, done, info = env_eval.step(act)
    cum_done = np.logical_or(cum_done, done)
    cum_rew += rew

    print(f'[t={t}] cum_rew: {cum_rew}')

    if cum_done.all():
        break

print(f'Mean score: {np.clip(cum_rew, 0., 4.).mean() * 25.}')
env_eval.close()
```

