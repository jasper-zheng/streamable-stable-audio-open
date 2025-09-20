# Streaming Stable Audio Open 1.0's Autoencoder

Streaming pre-trained [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)'s autoencoder with cached convolution, for realtime continuous inference. And scripting it to TorchScript to be used with [nn~](https://github.com/acids-ircam/nn_tilde) in MaxMSP/PureData.

> **Important note:** This doesn't stream the text-to-audio diffusion model in Stable Audio Open, this is only for the autoencoder (the pretransform model), the purpose is to use it for realtime latent manipulation or some other downstream tasks.



https://github.com/user-attachments/assets/ba8006f7-f559-481d-b7a0-5337bf483b2b



## Supported Pre-Trained Model

`stabilityai/stable-audio-open-1.0`: [HuggingFace](https://huggingface.co/stabilityai/stable-audio-open-1.0).  

To download the pretrained `stable-audio-open-1.0` model, you'll need a HuggingFace account and agree to Stability AI's License Agreement which can be found in the link above.  

## How to export to TorchScript

Use the `export.py` script to export the Stable Audio Open autoencoder to TorchScript format. The script provides several options to customize the export process.

**Important:** Make sure to add the `--streaming` flag for cached convolution, otherwise you will hear clicking artifacts when loaded in Max.   
**Important:** Buffer size in `nn~` is recommended to be something above `4096`, which can be set through the third argument in `nn~`.

```bash
python export.py --output path/to/exported/vae.ts --streaming
```

Choose the device for model loading and export using `--device`.
```bash
python export.py --device mps --streaming
```

Export the model in half precision (float16) adding `--half`, to potentially reduce inference speed.
```bash
python export.py --half --streaming
```

Skip the Gaussian noise in the encoder's variational reparametrization layer by adding `--skip-bottleneck`.
```bash
python export.py --skip-bottleneck --streaming
```

Test the scripted model with chunked audio by adding `--test`.
```bash
python export.py --test --streaming
```


## Disclaimer

This is a third party implementation and not made by Stability AI, for non-commercial research purpose. Please follow the [stable-audio-community](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md) license that comes with stable-audio-open-1.0.

## Acknowledgement

Reused model loading/building code from [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools.git)  
Reused exporting code from [nn_tilde](https://github.com/acids-ircam/nn_tilde.git)

