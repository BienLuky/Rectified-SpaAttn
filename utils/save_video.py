import os
import torchvision.transforms as T
import torch
import torchvision
import imageio
import numpy as np
import random
from einops import rearrange


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    transform = T.Compose([
        T.ToTensor(),  
    ])
    videos = [transform(img) for img in videos]  # List of [3, H, W]
    videos = torch.stack(videos, dim=1)  # shape: [3, T, H, W]
    videos = videos.unsqueeze(0)         # shape: [1, 3, T, H, W]

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, codec='libx264')

