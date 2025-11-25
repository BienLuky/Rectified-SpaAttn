# Image Metrics Evaluation

## Installation

We recommend using conda for enviornment management.

```bash
# create a virtual env and activate
conda create -n metrics python==3.10 
conda activate metrics

# install torch
pip install torch torchvision torchaudio

#install the requirements
pip install -r requirements.txt
```
## Evaluation of Rectified SpaAttn

We first prepare prompts from COCO annotations as `coco_1024.txt`.
And then we generate images according to the prompts.
Finally, we calculate metrics based on the image generated.


### Generate Image
```bash
bash inference.sh # including Flux.1-dev
```

### Calculate Metrics

### Introduction
This folder shows how we evaluate the generated images. We choose：
- *Clipscore* for text-image alignment
- *ImageReward* for human preference. 
- *FPFID* to measure the difference between the images generated with FP16 and the images generated with quantized model.

### Prepare Weights
Download the `ViT-L-14.pt` from this [Link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt), and place it under the `~/.cache/metrics_models`.

Download the `ImageReward.pt` and `med_config.json` on https://huggingface.co/THUDM/ImageReward/tree/main, and place it under the `~/.cache/metrics_models`.


### Easy Evaluation

```bash
bash evaluation.sh
```
