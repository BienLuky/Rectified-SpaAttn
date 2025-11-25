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
If you find [Rectified SpaAttn](https://arxiv.org/abs/2511.19835) is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```BibTeX
@misc{liu2025rectifiedspaattnrevisitingattention,
      title={Rectified SpaAttn: Revisiting Attention Sparsity for Efficient Video Generation}, 
      author={Xuewen Liu and Zhikai Li and Jing Zhang and Mengjuan Chen and Qingyi Gu},
      year={2025},
      eprint={2511.19835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19835}, 
}
```
