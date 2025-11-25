import tqdm
import os
import torch
import numpy as np
import random


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


def read_prompt_list(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def generate_func_flux(pipeline_image, pipeline, device_id, prompt_list, output_dir, loop: int = 5, kwargs_image: dict = {}, kwargs: dict = {}):
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            set_seed(l)
            pipeline_image.to(f"cuda:{device_id}")
            control_image = pipeline_image(prompt, **kwargs_image).images[0]
            control_image = control_image.resize((kwargs["width"], kwargs["height"]))
            pipeline_image.to("cpu")

            pipeline.to(f"cuda:{device_id}")
            kwargs["control_image"] = control_image
            image = pipeline(prompt, **kwargs).images[0]
            image.save(os.path.join(output_dir, f"{prompt}.png"))
            pipeline.to("cpu")




