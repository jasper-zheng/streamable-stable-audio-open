# from: https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools/models


import numpy as np
import random 

import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError


def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        latents = torch.randn_like(mean) * stdev + mean
        return latents

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x):

        mean, scale = x.chunk(2, dim=1)
        
        x = vae_sample(mean, scale)
        return x

    def decode(self, x):
        return x

def compute_mean_kernel(x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

def compute_mmd(latents):
    latents_reshaped = latents.permute(0, 2, 1).reshape(-1, latents.shape[1])
    noise = torch.randn_like(latents_reshaped)

    latents_kernel = compute_mean_kernel(latents_reshaped, latents_reshaped)
    noise_kernel = compute_mean_kernel(noise, noise)
    latents_noise_kernel = compute_mean_kernel(latents_reshaped, noise)
    
    mmd = latents_kernel + noise_kernel - 2 * latents_noise_kernel
    return mmd.mean()

class WassersteinBottleneck(Bottleneck):
    def __init__(self, noise_augment_dim: int = 0, bypass_mmd: bool = False, use_tanh: bool = False, tanh_scale: float = 5.0):
        super().__init__(is_discrete=False)

        self.noise_augment_dim = noise_augment_dim
        self.bypass_mmd = bypass_mmd
        self.use_tanh = use_tanh
        self.tanh_scale = tanh_scale
    
    def encode(self, x, return_info=False):
        info = {}

        if self.training and return_info:
            if self.bypass_mmd:
                mmd = torch.tensor(0.0)
            else:
                mmd = compute_mmd(x)
                
            info["mmd"] = mmd

        if self.use_tanh:
            x = torch.tanh(x / self.tanh_scale) * self.tanh_scale
        
        if return_info:
            return x, info
        
        return x

    def decode(self, x):

        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x

     
    