# SVD-webui
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Main Repo 

Visit the following links for the details of Stable Video Diffusion.

Codebase: https://github.com/Stability-AI/generative-models

HF: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid

LICENSE: [STABLE VIDEO DIFFUSION NON-COMMERCIAL COMMUNITY LICENSE AGREEMENT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/LICENSE)

Paper: https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets

Code refence

[@mk1stats](https://twitter.com/mk1stats)

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb)

## Usage

### Clone repo with submodules
1、
```
git clone --recurse-submodules https://github.com/sdbds/SVD-webui/
```

OR

1、Download code zip from Release. 

2、
```
git submodule update --recursive --init
```

### Required Dependencies

- Python 3.10.6 ~ 3.10.11
- Git

### Windows

#### Installation

Run `install.ps1` or `install_cn.ps1` will automaticilly create a venv for you and install necessary deps.

#### Download Model
svd: [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors?download=true) for 14 frames generation NEED 15GB VRAM

svd_xt: [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true) for 25 frames generation NEED 18GB VRAM

#### Run GUI

Edit `run_gui.ps1` for change model version or outputs dir.
```
$model_path="./checkpoints/svd_xt.safetensors"
$outputs="./outputs"
```

Run `run_gui.ps1` will get a address,open browser with it.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

****
