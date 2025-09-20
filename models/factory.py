# from: https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools/models


import json

def create_pretransform_from_config(pretransform_config, sample_rate, skip_bottleneck, device):
    pretransform_type = pretransform_config.get('type', None)

    assert pretransform_type is not None, 'type must be specified in pretransform config'

    if pretransform_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        from .pretransforms import AutoencoderPretransform

        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_config["config"]}
        
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)

        pretransform = AutoencoderPretransform(autoencoder, scale=scale, model_half=model_half, skip_bottleneck=skip_bottleneck, test_device=device)
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')
    
    enable_grad = pretransform_config.get('enable_grad', False)
    pretransform.enable_grad = enable_grad

    pretransform.eval().requires_grad_(pretransform.enable_grad)

    return pretransform

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'vae':
        from .bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    elif bottleneck_type == "wasserstein":
        from .bottleneck import WassersteinBottleneck
        bottleneck = WassersteinBottleneck(**bottleneck_config.get("config", {}))
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')
    
    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck
