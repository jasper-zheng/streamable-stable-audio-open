# Taken from https://github.com/zoli333/Weight-Normalization/blob/master/layers.py

from enum import auto, Enum
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.utils import parametrize

from torch import linalg as LA

class _WeightNorm(Module):
    def __init__(
        self,
        dim: Optional[int] = 0,
    ) -> None:
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim

    @staticmethod
    def norm_except_dim_0(weight):
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        out = LA.norm(weight.view(weight.size(0), -1), ord=2, dim=1).view(*output_size)
        return out
    
    def forward(self, weight_g, weight_v):
        # implementation of weight normalization:
        w = weight_g * weight_v / self.norm_except_dim_0(weight_v)
        return w

    def right_inverse(self, weight):
        weight_g = torch.norm_except_dim(weight, 2, self.dim)
        weight_v = weight

        return weight_g, weight_v

def weight_norm(module: Module, name: str = "weight", dim: int = 0):
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` with two parameters: one specifying the magnitude
    and one specifying the direction.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    """
    _weight_norm = _WeightNorm(dim)
    parametrize.register_parametrization(module, name, _weight_norm, unsafe=True)

    def _weight_norm_compat_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        g_key = f"{prefix}{name}_g"
        v_key = f"{prefix}{name}_v"
        if g_key in state_dict and v_key in state_dict:
            original0 = state_dict.pop(g_key)
            original1 = state_dict.pop(v_key)
            state_dict[f"{prefix}parametrizations.{name}.original0"] = original0
            state_dict[f"{prefix}parametrizations.{name}.original1"] = original1

    module._register_load_state_dict_pre_hook(_weight_norm_compat_hook)
    return module
