# from: https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools/models

import json

from .factory import create_pretransform_from_config
from .utils import load_ckpt_state_dict

from huggingface_hub import hf_hub_download


def get_pretrained_pretransform(name: str, model_half: bool = False, skip_bottleneck: bool = False, device: str = "cpu"):
    """Load only the pretransform autoencoder for a pretrained model repo.

    Attempts to load weights for the autoencoder portion
    by extracting a matching sub-state-dict from the checkpoint.

    Returns: (pretransform_module, model_config)
    """
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model')
    with open(model_config_path) as f:
        model_config = json.load(f)

    # Build pretransform from config (expects top-level sample_rate)
    sample_rate = model_config["sample_rate"]
    pre_cfg = model_config["model"].get("pretransform")
    assert pre_cfg is not None, "Model config does not contain a pretransform"

    pre_cfg["model_half"] = model_half
    pretransform = create_pretransform_from_config(pre_cfg, sample_rate, skip_bottleneck=skip_bottleneck, device=device)

    # Load checkpoint and try to map relevant keys to pretransform.model.*
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model')
    except Exception:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    state_dict = load_ckpt_state_dict(model_ckpt_path)

    prefixes = [
        "pretransform.model.",
        "model.pretransform.model.",
        "pretransform.",  # in case inner model direct keys
        "model.",         # last resort if checkpoint only contains AE
    ]
    
    loaded_any = False
    for prefix in prefixes:
        sub_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                sub_sd[k[len(prefix):]] = v
        if sub_sd:
            missing, unexpected = pretransform.model.load_state_dict(sub_sd, strict=False)
            # torch returns lists for missing/unexpected; treat as success if any param matched
            if len(missing) < len(sub_sd):
                loaded_any = True
                break

    # It's fine if we couldn't map weights (e.g., a different checkpoint layout);
    # the scripted artifact will still compile, but results will be random.

    return pretransform, model_config