import torch
from einops import rearrange
from torch import nn
from torchaudio.transforms import Resample

from .nn_tilde import Module

class Pretransform(Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()
        
        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, skip_bottleneck=False, test_device="cpu"):
        super().__init__(enable_grad=False, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self._methods = []
        self._attributes = ["none"]
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half

        self.encoded_channels = model.latent_dim

        self.skip_bottleneck = skip_bottleneck

        if self.model_half:
            self.model.half()

        self.to(test_device)
        self.eval()

        self.register_method(
            "forward",
            in_channels=2,
            in_ratio=1,
            out_channels=2,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, 1 + 2)],
            output_labels=['(signal) Channel %d'%d for d in range(1, 1+2)],
            test_method=False,
            test_buffer_size = 4096,
            test_device=test_device
        )

        self.register_method(
            "encode",
            in_channels=2,
            in_ratio=1,
            out_channels=model.latent_dim,
            out_ratio=model.downsampling_ratio,
            input_labels=['(signal) Channel %d'%d for d in range(1, 1 + 2)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(model.latent_dim)
            ],
            test_method=False,
            test_buffer_size = 4096,
            test_device=test_device
        )

        self.register_method(
            "decode",
            in_channels=model.latent_dim,
            in_ratio=model.downsampling_ratio,
            out_channels=2,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(model.latent_dim)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, 1+2)],
            test_method=False,
            test_buffer_size = 4096,
            test_device=test_device
        )
    
    @torch.jit.export
    def encode(self, x):
        # x: (batch_size, audio_channels, sample)
        
        if self.model_half:
            x = x.half()

        encoded = self.model.encode(x, skip_bottleneck=self.skip_bottleneck)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale # (batch_size, latent_dim, sample)

    @torch.jit.export
    def decode(self, z):
        # z: (batch_size, latent_dim, sample)

        z = z * self.scale

        if self.model_half:
            z = z.half()

        decoded = self.model.decode(z, skip_bottleneck=self.skip_bottleneck)

        if self.model_half:
            decoded = decoded.float()

        return decoded # (batch_size, audio_channels, sample)
    
    @torch.jit.export
    def forward(self, x):
        # x: (batch_size, audio_channels, sample)
        if self.model_half:
            x = x.half()
        z = self.model.encode(x, skip_bottleneck=self.skip_bottleneck)
        y = self.model.decode(z, skip_bottleneck=self.skip_bottleneck)
        if self.model_half:
            y = y.float()
        return y # (batch_size, audio_channels, sample)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

    def to_half(self):

        self.model.half()
        self.model_half = True
        return self
