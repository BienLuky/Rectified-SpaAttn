<p align="center">
  <img src="./assets/title.png" width="80%" max-width="800px">
</p>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Rectified_SpaAttn-2511.19835-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.19835)
[![GitHub Stars](https://img.shields.io/github/stars/BienLuky/Rectified-SpaAttn.svg?style=social&label=Star&maxAge=60)](https://github.com/BienLuky/Rectified-SpaAttn)
 <br>



## 🔥 News
- [2025/12] Rectified SpaAttn is extended to CogVideoX1.5, achieving a 3× acceleration.
- [2025/11] Rectified SpaAttn is open-sourced! HunyuanVideo, Wan2.1, and Flux.1-dev can be accelerated by 4-9×.

## 📖 Overview
Rectified SpaAttn rectifies sparse attention allocation with implicit full attention reference, thereby enhancing the alignment between sparse and full attention maps.
<p align="center">
  <img src="./assets/overview.png" width="100%" max-width="800px">
</p>

## 📃 Open-source Plan

- Model Adaptation
  - [ ] Your models ☺️ 
  - [ ] Flux.2 
  - [ ] Wan2.2 (I2V / T2V)
  - [x] CogVideoX1.5 (I2V / T2V)
  - [x] HunyuanVideo, Wan2.1 (I2V / T2V), Flux.1-dev
- Engineering Optimization
  - [ ] FP8 Attention
  - [ ] RoPE & Norm Kernel
  - [ ] FA3 Adaptation

We warmly welcome and sincerely appreciate contributions from the community to further enhance Rectified SpaAttn. 
If your generative model is hindered by attention latency, please feel free to contact us—we are more than willing to help integrate Rectified SpaAttn into your model.

## 🛠️ Installation
Rectified SpaAttn features a straightforward installation procedure with the following two steps:

```bash
# 1. Repository Cloning
git clone https://github.com/BienLuky/Rectified-SpaAttn.git
cd Rectified-SpaAttn

# 2. Conda Environment Setup 
conda create -n rectified python==3.10.19 
conda activate rectified

pip install PyTorch # We recommend using CUDA versions 12.4 / 12.8 + PyTorch versions 2.5.1 / 2.6.0
pip install -r requirements # Install other packages
pip install flash-attn --no-build-isolation

```

## 🚀 Inference Examples

### HunyuanVideo

The running scripts are:
```bash
python scripts/main_hunyuan.py  # Rectified SpaAttn with 2.50× speedup
python scripts/main_hunyuan.py --enable_teacache # Rectified SpaAttn+Cache with 5.24× speedup
```

### Wan 2.1
The running scripts are:
```bash
# Text-to-Video
python scripts/main_wan21t2v.py # Rectified SpaAttn with 1.68× speedup
python scripts/main_wan21t2v.py --enable_teacache # Rectified SpaAttn+Cache with 4.61× speedup

# Image-to-Video
python scripts/main_wan21i2v.py # Rectified SpaAttn with 1.81× speedup
python scripts/main_wan21i2v.py --enable_teacache # Rectified SpaAttn+Cache with 8.97× speedup
```

### Flux.1-dev

The running scripts are:
```bash
python scripts/main_upflux.py  # Rectified SpaAttn with 1.60× speedup
python scripts/main_upflux.py --enable_teacache # Rectified SpaAttn+Cache with 4.15× speedup
```

### CogVideoX1.5
The running scripts are:
```bash
# Text-to-Video
python scripts/main_cogvideox.py --generate_type t2v --sa_drop_rate 0.85 # Rectified SpaAttn with 1.76× speedup
python scripts/main_cogvideox.py --generate_type t2v --sa_drop_rate 0.85 --enable_teacache # Rectified SpaAttn+Cache with 2.97× speedup

# Image-to-Video
python scripts/main_cogvideox.py --generate_type i2v --sa_drop_rate 0.75 # Rectified SpaAttn with 1.60× speedup
python scripts/main_cogvideox.py --generate_type i2v --sa_drop_rate 0.75 --enable_teacache # Rectified SpaAttn+Cache with 2.90× speedup
```

## 📊 Evaluation Pipeline

We provide a complete evaluation pipeline — including tool preparation, model generation, and result assessment — all organized within the `eval` directory.


## 🎥 Demo

### HunyuanVideo (128 frames, 720p)
<table>
  <tr>
    <td align="center">
      <img src="assets/hunyuan1_full-min.gif" width="100%"/><br>
      <em>Dense Attention (<strong>2430 s</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/hunyuan1_ours-min.gif" width="100%"/><br>
      <em>Ours (<strong>729 s</em> ⬆️ <em>3.33×</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/hunyuan1_ourstea-min.gif" width="100%"/><br>
      <em>Ours (<strong>463 s</em> ⬆️ <em>5.24×</strong>)</em>
    </td>
  </tr>
</table>
<p align="center">
  <strong>prompt:</strong>
  <em>"several hot air balloons flying over a city."</em><br>
</p>

<div style="margin-top: 25px;"></div>

### Wan2.1-T2V (81 frames, 720p)
<table>
  <tr>
    <td align="center">
      <img src="assets/want2v2_full-min.gif" width="100%"/><br>
      <em>Dense Attention (<strong>2731 s</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/want2v2_ours-min.gif" width="100%"/><br>
      <em>Ours (<strong>1515 s</em> ⬆️ <em>1.80×</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/want2v2_ourstea-min.gif" width="100%"/><br>
      <em>Ours (<strong>592 s</em> ⬆️ <em>4.61×</strong>)</em>
    </td>
  </tr>
</table>
<p align="center">
  <strong>prompt:</strong>
  <em>"A sleek white yacht gliding across a crystal-blue sea at sunset, camera circles the vessel as golden light sparkles on gentle waves, slight lens distortion."</em><br>
</p>

<div style="margin-top: 25px;"></div>

### Wan2.1-I2V (81 frames, 720p)
<table>
  <tr>
    <td align="center">
      <img src="assets/wani2v2_full-min.gif" width="100%"/><br>
      <em>Dense Attention (<strong>2754 s</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/wani2v2_ours-min.gif" width="100%"/><br>
      <em>Ours (<strong>1327 s</em> ⬆️ <em>2.08×</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/wani2v2_ourstea-min.gif" width="100%"/><br>
      <em>Ours (<strong>307 s</em> ⬆️ <em>8.97×</strong>)</em>
    </td>
  </tr>
</table>
<p align="center">
  <strong>prompt:</strong>
  <em>"a boat sits on the shore of a lake with mt fuji in the background."</em><br>
</p>

<div style="margin-top: 25px;"></div>

### Flux.1-dev (4096 × 4096 Resolution)
<table>
  <tr>
    <td align="center">
      <img src="assets/fluxup1_full-min.png" width="100%"/><br>
      <em>Dense Attention (<strong>912 s</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/fluxup1_ours-min.png" width="100%"/><br>
      <em>Ours (<strong>570 s</em> ⬆️ <em>1.60×</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/fluxup1_ours+tea-min.png" width="100%"/><br>
      <em>Ours (<strong>220 s</em> ⬆️ <em>4.15×</strong>)</em>
    </td>
  </tr>
</table>
<p align="center">
  <strong>prompt:</strong>
  <em>"Mountain landscape with a wooden sign reading Rectified SpaAttn."</em><br>
</p>

## 📚 Citation

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

## 💙 Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Flux.1-dev](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler), [Jenga](https://github.com/dvlab-research/Jenga), [TeaCache](https://github.com/ali-vilab/TeaCache) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.

