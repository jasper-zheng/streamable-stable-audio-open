"""
Export the Stable Audio Open 1.0 autoencoder (pretransform) to TorchScript.

https://huggingface.co/stabilityai/stable-audio-open-1.0

"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
torch_250 = True if torch.__version__ >= "2.5" else False

from models import get_pretrained_pretransform

import cached_conv as cc

def pick_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def remove_parametrizations(module: torch.nn.Module) -> None:
    try:
        from torch.nn.utils import parametrize as _parametrize
    except Exception:
        return
    for m in module.modules():
        if hasattr(m, "parametrizations"):
            names = list(getattr(m, "parametrizations").keys())
            for pname in names:
                try:
                    _parametrize.remove_parametrizations(m, pname, leave_parametrized=True)
                except Exception:
                    pass

def test_streaming(pretransform: torch.nn.Module, device: str) -> None:
    print("Testing the exported model...")
    # Test the exported model with chunked audio
    import librosa, time
    audio_path = librosa.example('trumpet')
    wv, sr = librosa.load(audio_path, sr=44100, mono=False)
    wv = torch.tensor(wv, device=device).unsqueeze(0)[:,:4096*20].repeat(2,1).unsqueeze(0)  # make stereo, limit length for test
    wv_chunks = [wv[:, :, i*4096:(i+1)*4096] for i in range(20)]
    print(f'waveform shape: {wv.shape}')
    print(f'number of chunks: {len(wv_chunks)}')
    print(f'chunk shape: {wv_chunks[0].shape}')

    print(f'Running encoder, test device: {device}')
    ## Run audio chunks to the encoder

    pretransform = pretransform.to(device)

    latent_chunks = []
    with torch.no_grad():
        if torch_250:
            torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
            start_time = time.perf_counter()
        for i, w in enumerate(wv_chunks):
            latent = pretransform.encode(w)
            latent_chunks.append(latent)
        if torch_250:
            torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
            end_time = time.perf_counter()
            print(f'Encoder execution time: {end_time - start_time:.2f} seconds')
            

    print(f'Running decoder, test device: {device}')
    ## Run audio chunks to the decoder
    wv_recons = []
    with torch.no_grad():
        if torch_250:
            torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
            start_time = time.perf_counter()
        for i, latent in enumerate(latent_chunks):
            wv_recon = pretransform.decode(latent)
            wv_recons.append(wv_recon)
        if torch_250:
            torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
            end_time = time.perf_counter()
            print(f'Decoder execution time: {end_time - start_time:.2f} seconds')
    wv_recon = torch.cat(wv_recons, dim=-1)
    print(f'reconstructed waveform shape: {wv_recon.shape}')

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export Stable Audio pretransform (autoencoder) to TorchScript")
    p.add_argument("--pretrained-name", default="stabilityai/stable-audio-open-1.0",
                   help="HuggingFace model repo id that provides model_config.json and weights")
    p.add_argument("--output", default="exports/stable-vae.ts", help="Path to write TorchScript file (.ts)")
    p.add_argument("--export-device", choices=["cpu", "cuda", "mps"], default=None, help="Save on cpu/cuda/mps? (default: auto)")
    p.add_argument("--test-device", choices=["cpu", "cuda", "mps"], default=None, help="Test on cpu/cuda/mps? (default: auto)")
    p.add_argument("--half", action="store_true", help="Run the autoencoder in half precision")
    p.add_argument("--streaming", action="store_true", help="Enable cached convolution for streaming")
    p.add_argument("--test", action="store_true", help="Testing the exported model with chunked audio")
    p.add_argument("--skip-bottleneck", action="store_true", help="Skip the variational reparametrization in the VAE bottleneck")
    p.add_argument("--keep-parametrizations", action="store_true",
                   help="Keep torch parametrizations when exporting, which will likely to fail.")

    args = p.parse_args(argv)

    cc.use_cached_conv(args.streaming)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_device = pick_device(args.export_device)
    test_device = pick_device(args.test_device)
    print(f"Using device for export: {export_device}, torch version: {torch.__version__}")

    # Prefer lightweight path: load only the pretransform autoencoder
    pretransform, model_config = get_pretrained_pretransform(args.pretrained_name, model_half=args.half, skip_bottleneck=args.skip_bottleneck, device=test_device)
    assert pretransform is not None

    print(f"sample_rate: {model_config.get('sample_rate', 'unknown')}")
    print(f"latent_dim: {model_config['model']['pretransform']['config'].get('latent_dim', 'unknown')}")
    print(f"downsampling_ratio/compression_ratio: {model_config['model']['pretransform']['config'].get('downsampling_ratio', 'unknown')}")
    print(f"io_channels: {model_config['model']['pretransform']['config'].get('io_channels', 'unknown')}")
    
    pretransform.eval()
    
    # Remove parametrizations of weight_norm
    if not args.keep_parametrizations:
        print("Removing parametrizations.")
        remove_parametrizations(pretransform)

    pretransform=pretransform.to(test_device)
    x = torch.zeros(2, pretransform.io_channels, 8192).to(test_device)
    print("Testing the model with silence")
    with torch.no_grad():
        y = pretransform.forward(x)
    print(f"Test run successful, output shape: {y.shape}")

    pretransform = pretransform.to(export_device)
    print(f"Exporting TorchScript to: {out_path}")
    scripted = pretransform.export_to_ts(str(out_path))

    print("Done.")
    if not args.test:
        return 0
    
    # Test the exported model with chunked audio
    test_streaming(scripted, test_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
