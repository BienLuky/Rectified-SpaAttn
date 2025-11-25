import json
import tqdm
import os
import torchvision.transforms as T
import torch
import torchvision
import numpy as np
import random
import imageio
from einops import rearrange


def set_seed(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


def read_image_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    image_list = [prompt["image_path"] for prompt in prompt_list]
    return image_list


def generate_func_hunyuan(pipeline, prompt_list, output_dir, loop: int = 5, kwargs: dict = {}):
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            set_seed(l)
            video = pipeline(prompt, **kwargs).frames[0]
            save_videos_grid(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"), fps=24)    


def generate_func_wan21t2v(pipeline, prompt_list, output_dir, loop: int = 5, kwargs: dict = {}):
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            set_seed(l)
            video = pipeline(prompt, **kwargs).frames[0]
            save_videos_grid(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"), fps=16)    
        prompt_id += 1


def generate_func_wan21i2v(pipeline, prompt_list, image_list, output_dir, loop: int = 5, kwargs: dict = {}):
    for i in tqdm.tqdm(range(len(prompt_list))):
        prompt = prompt_list[i]
        image = image_list[i]
        for l in range(loop):
            set_seed(l)
            video = pipeline(image, prompt, **kwargs).frames[0]
            save_videos_grid(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"), fps=16)    



