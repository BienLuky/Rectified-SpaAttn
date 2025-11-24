<p align="center">
  <img src="./assets/title.png" width="80%" max-width="800px">
</p>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/DilateQuant-2409.14307-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2409.14307)
[![GitHub Stars](https://img.shields.io/github/stars/BienLuky/DilateQuant.svg?style=social&label=Star&maxAge=60)](https://github.com/BienLuky/DilateQuant)
 <br>

</h5>

> The offical implementation of the paper [**Training-Free Efficient Video Generation via Dynamic Token Carving**](https://arxiv.org/abs/2505.16864) <be>
</h5>


# Visualization-of-Rectified-SpaAttn

The project intended to showcase the visualization results of Rectified SpaAttn.



## 🎥 Demo

### HunyuanVideo (128 frames, 720p)
<table>
  <tr>
    <td align="center">
      <img src="assets/hunyuan1_full.gif" width="100%"/><br>
      <em>Dense Attention (<strong>41 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/hunyuan1_ours.gif" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>12 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/hunyuan1_ours+tea.gif" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>8 min</strong>)</em>
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
      <img src="assets/want2v2_full.gif" width="100%"/><br>
      <em>Dense Attention (<strong>46 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/want2v2_ours.gif" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>25 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/want2v2_ours+tea.gif" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>10 min</strong>)</em>
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
      <img src="assets/wani2v1_full.gif" width="100%"/><br>
      <em>Dense Attention (<strong>46 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/wani2v1_ours.gif" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>22 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/wani2v1_ours+tea.gif" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>5 min</strong>)</em>
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
      <em>Dense Attention (<strong>15 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/fluxup1_ours-min.png" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>9 min</strong>)</em>
    </td>
    <td align="center">
      <img src="assets/fluxup1_ours+tea-min.png" width="100%"/><br>
      <em>Rectified SpaAttn (<strong>4 min</strong>)</em>
    </td>
  </tr>
</table>
<p align="center">
  <strong>prompt:</strong>
  <em>"Mountain landscape with a wooden sign reading Rectified SpaAttn."</em><br>
</p>

