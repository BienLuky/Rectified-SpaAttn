# Video Metrics Evaluation

## Installation

We strongly recommend using Anaconda to create a new environment (Python >= 3.10) to run our examples:

```bash
conda create -n video_eval python=3.10
conda activate video_eval
pip install -r video_requirements.txt  # install VBench packages
pip install -r VisionReward/requirements.txt # install VR packages
```


## Evaluation of Rectified SpaAttn

We first prepare prompts from PenguinVideoBenchmark.csv and prompt-image pairs from Vbench.
And then we generate videos according to the prompts.
Finally, we calculate Vbench and VR Scores based on the video generated.

### Prepare Data
```bash
# Text-to-Video Prompts
python vbench/get_prompts.py

# Image-to-Video Prompt-Image Pairs
# 1. Download the initial VBench prompt–image pairs from the following link: https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx?usp=sharing
# and place them in ./vbench/origin.

# 2. Center-crop the images to the appropriate 16:9 aspect ratio.
python vbench/crop_image.py

# 3. Construct the JSON file containing the prompt–image pairs.
python vbench/get_prompt-image_pair.py
```

### Generate video
```bash
bash inference.sh # including HunyuanVideo, Wan 2.1 T2V / I2V
```

### Calculate Vbench and VR Score
```bash
bash evaluation.sh
```


## Citation
If you find Rectified SpaAttn is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```
