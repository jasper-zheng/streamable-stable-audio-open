# from: https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools/models

import torch
import math
import numpy as np
import math

from torch import nn, sin, pow
from torch.nn import functional as F
try:
    from torch.nn.utils.parametrizations import weight_norm
    pt250 = True
except ImportError:
    from .weight_norm import weight_norm
pt250 = False
from torchaudio import transforms as T
from alias_free_torch import Activation1d
from typing import List, Literal, Dict, Any, Callable, cast, Optional
from einops import rearrange

from .blocks import SnakeBeta
from .bottleneck import Bottleneck
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform, AutoencoderPretransform

# Optional streamable convs (cached padding mechanism)
try:
    import cached_conv as _cc  # type: ignore
    HAS_CC = True
    cc = cast(Any, _cc)
except Exception:
    cc = cast(Any, None)  # type: ignore
    HAS_CC = False

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Streamable conv helpers
def _streamable_conv1d(in_channels: int, out_channels: int, kernel_size: int,
                       stride: int = 1, padding: tuple[int, int] | int | str | None = None,
                       dilation: int = 1, bias: bool = True, streaming: bool = False):
    """
    Returns a weight-normalized Conv1d. If streaming and cached_conv is available,
    uses cc.Conv1d with cached padding to match non-causal receptive field in batch mode
    while enabling causal streaming execution.
    """
    if streaming and HAS_CC:
        # Use cached padding computation to emulate same receptive field
        if padding is None:
            pad = cc.get_padding(kernel_size, stride=stride, dilation=dilation)
        else:
            pad = padding
        m = cc.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=bias,
        )
        return weight_norm(m)
    else:
        if isinstance(padding, str):
            # e.g., 'same' when used with nearest upsample path
            m = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,  # type: ignore[arg-type]
                dilation=dilation,
                bias=bias,
            )
        else:
            pad = 0 if padding is None else (padding if isinstance(padding, int) else 0)
            m = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=bias,
            )
        return weight_norm(m)


def _streamable_conv_transpose1d(in_channels: int, out_channels: int, kernel_size: int,
                                 stride: int = 1, padding: int | None = None,
                                 bias: bool = True, streaming: bool = False):
    """
    Returns a weight-normalized ConvTranspose1d. If streaming and cached_conv is available,
    uses cc.ConvTranspose1d. Padding heuristic mirrors non-streaming path for shape parity.
    """
    if streaming and HAS_CC:
        # cached_conv transpose uses similar API; use same padding heuristic
        pad = math.ceil(stride / 2) if padding is None else padding
        m = cc.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=bias,
        )
        return weight_norm(m)
    else:
        pad = 0 if padding is None else padding
        m = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=bias,
        )
        return weight_norm(m)

from torch.utils.checkpoint import checkpoint as torch_checkpoint

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        act = Activation1d(act)
    
    return act

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

def _reset_cached_conv_state(module: nn.Module) -> None:
    """Recursively reset cached_conv streaming caches if present."""
    if not HAS_CC:
        return
    for m in module.modules():
        if hasattr(m, "reset_caches") and callable(getattr(m, "reset_caches")):
            try:
                m.reset_caches()  # type: ignore[attr-defined]
            except Exception:
                pass

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False, streaming: bool = False):
        super().__init__()

        self.dilation = dilation
        self.streaming = bool(streaming and HAS_CC)

        padding = (dilation * (7 - 1)) // 2

        p = (7 - 1) * dilation + 1
        p_right = p // 2
        p_left = (p - 1) // 2

        if self.streaming:
            # Build streamable path using cached padding ops and residual alignment
            net = [
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
                _streamable_conv1d(in_channels, out_channels, kernel_size=7, dilation=dilation,
                                   stride=1, padding=(p_left, p_right), streaming=True),
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
                _streamable_conv1d(out_channels, out_channels, kernel_size=1, streaming=True),
            ]
            self.layers = cc.CachedSequential(*net)
            # Align residual with main branch according to accumulated delay
            additional_delay = self.layers[1].cumulative_delay
            self.aligned = cc.AlignBranches(self.layers, nn.Identity(), delays=[additional_delay, 0])

        else:
            self.layers = nn.Sequential(
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
                WNConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=7, dilation=dilation, padding=padding),
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
                WNConv1d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=1),
            )

    def forward(self, x):
        if self.streaming:
            x_net, x_res = self.aligned(x)
            return x_net + x_res
        else:
            res = x
            x = self.layers(x)
            return x + res

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return rearrange(x, '... a b -> ... b a')


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, streaming: bool = False):
        super().__init__()

        if streaming and HAS_CC:
            layers = [
                ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake, streaming=True),
                ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake, streaming=True),
                ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake, streaming=True),
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
                _streamable_conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, streaming=True),
            ]
            self.layers = cc.CachedSequential(*layers)
        else:
            self.layers = nn.Sequential(
                ResidualUnit(in_channels=in_channels,
                             out_channels=in_channels, dilation=1, use_snake=use_snake),
                ResidualUnit(in_channels=in_channels,
                             out_channels=in_channels, dilation=3, use_snake=use_snake),
                ResidualUnit(in_channels=in_channels,
                             out_channels=in_channels, dilation=9, use_snake=use_snake),
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
                WNConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False, streaming: bool = False):
        super().__init__()

        if streaming and HAS_CC:
            if use_nearest_upsample:
                upsample_layer = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="nearest"),
                    _streamable_conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=1, padding='same', streaming=True, bias=False),
                )
            else:
                upsample_layer = _streamable_conv_transpose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    streaming=True,
                    bias=True,
                )

            layers = [
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
                upsample_layer,
                ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake, streaming=True),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake, streaming=True),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake, streaming=True),
            ]
            self.layers = cc.CachedSequential(*layers)
        else:
            if use_nearest_upsample:
                upsample_layer = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="nearest"),
                    WNConv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=2 * stride,
                             stride=1,
                             bias=False,
                             padding='same')
                )
            else:
                upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))

            self.layers = nn.Sequential(
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
                upsample_layer,
                ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                             dilation=1, use_snake=use_snake),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                             dilation=3, use_snake=use_snake),
                ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                             dilation=9, use_snake=use_snake),
            )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 streaming: bool = False
        ):
        super().__init__()
        self.in_channels = in_channels
        self.streaming = bool(streaming and HAS_CC)
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        p = (7 - 1) * 1 + 1
        p_right = p // 2
        p_left = (p - 1) // 2

        if self.streaming:
            layers = [
                _streamable_conv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, stride=1, streaming=True, padding=(p_left, p_right))
            ]
        else:
            layers = [
                WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
            ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake, streaming=self.streaming)]

        if self.streaming:
            layers += [
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
                _streamable_conv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, streaming=True),
            ]
            self.layers = cc.CachedSequential(*layers)
        else:
            layers += [
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
                WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
            ]

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_streaming_state(self) -> None:
        """Clear internal streaming caches when using cached_conv."""
        _reset_cached_conv_state(self)


class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True,
                 streaming: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.streaming = bool(streaming and HAS_CC)

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        if self.streaming:
            layers = [
                _streamable_conv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, streaming=True),
            ]
        else:
            layers = [
                WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
            ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels,
                out_channels=c_mults[i-1]*channels,
                stride=strides[i-1],
                use_snake=use_snake,
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample,
                streaming=self.streaming,
            )]

        if self.streaming:
            layers += [
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
                _streamable_conv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, streaming=True, bias=False),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
            self.layers = cc.CachedSequential(*layers)
        else:
            layers += [
                get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
                WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
                nn.Tanh() if final_tanh else nn.Identity()
            ]

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset_streaming_state(self) -> None:
        """Clear internal streaming caches when using cached_conv."""
        _reset_cached_conv_state(self)

class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Optional[Bottleneck] = None,
        pretransform: Optional[Pretransform] = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, skip_bottleneck: bool = False):
        if self.encoder is not None:
            with torch.no_grad():
                latents = self.encoder(audio)
        else:
            latents = audio

        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.encode(latents)
        else:
            latents, _ = latents.chunk(2, dim=1)
        return latents
            

    def decode(self, latents, skip_bottleneck: bool = False):

        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents)

        if self.pretransform is not None:
            with torch.no_grad():
                decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        
        return decoded

    # Streaming utilities
    def reset_streaming_state(self) -> None:
        """
        Reset cached-convolution internal state in encoder/decoder for clean streaming runs.
        Safe to call even if cached_conv is not installed or streaming was disabled.
        """
        if isinstance(self.encoder, nn.Module):
            _reset_cached_conv_state(self.encoder)
        if isinstance(self.decoder, nn.Module):
            _reset_cached_conv_state(self.decoder)
       
# AE factories

def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
            **encoder_config["config"], streaming=True
        )
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"], streaming=True
        )
    elif decoder_type == "local_attn":
        # Lazy import only if local attention decoder is requested
        from .local_attention import TransformerDecoder1D  # type: ignore

        local_attn_config = decoder_config["config"]

        decoder = TransformerDecoder1D(
            **local_attn_config
        )
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(config: Dict[str, Any]):
    
    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        
    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )
