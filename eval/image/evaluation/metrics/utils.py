'''
@File       :   utils.py
@Time       :   2023/04/05 19:18:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
* Based on CLIP code base
* https://github.com/openai/CLIP
* Checkpoint of CLIP/BLIP/Aesthetic are from:
* https://github.com/openai/CLIP
* https://github.com/salesforce/BLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from .ImageReward.ImageReward import ImageReward
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .models.CLIPScore import CLIPScore
from .models.BLIPScore import BLIPScore
from .models.AestheticScore import AestheticScore
from .align_sd.HPS import HPS
from selenium import webdriver
import time
import requests
from .PickScore.PickScore import PickScore

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, preprocessor):
        self.files = files[:100]
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        img = self.preprocessor(img)
        return img

def ImageReward_download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return download_target

def download_from_onedrive(onedrive_link, destination_path):
    # 创建 Chrome WebDriver 实例
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 无头模式，不显示浏览器界面
    driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)

    try:
        # 打开 OneDrive 页面
        driver.get(onedrive_link)
        time.sleep(5)  # 等待页面加载，根据需要调整等待时间

        # 模拟点击下载按钮
        download_button = driver.find_element_by_class_name('download-button')
        download_button.click()
        time.sleep(5)  # 等待下载过程，根据文件大小和网络速度调整等待时间

        # 获取当前页面的 URL（下载链接）
        download_link = driver.current_url

        # 使用获取到的直接下载链接下载文件
        download_response = requests.get(download_link)
        if download_response.status_code == 200:
            with open(destination_path, 'wb') as file:
                file.write(download_response.content)
            print(f"文件已下载到：{destination_path}")
        else:
            print("无法下载文件")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        # 关闭浏览器
        driver.quit()

_SCORES = {
    "CLIP": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "BLIP": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
    "Aesthetic": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
    "HPS": "~/.cache/metric_models/hpc.pt",
    "ImageReward": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
    "PickScore": ""
}


def available_scores() -> List[str]:
    """Returns the names of available scores"""
    return list(_SCORES.keys())


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def load_score(name: str = "CLIP", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None):
    """Load a metric model

    Parameters
    ----------
    name : str
        A model name listed by `available_models()`

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/metric_models/"

    Returns
    -------
    model : torch.nn.Module
        The model
    """
    model_download_root = download_root or os.path.expanduser("/home/lizhikai/.cache/metric_models")
    if name != 'PickScore' and name != "HPS":
        if name in _SCORES:
            file_name = _SCORES[name].split("/")[-1]
            if not os.path.exists(os.path.join(model_download_root, file_name)): 
                model_path = _download(_SCORES[name], model_download_root)
            else:
                model_path = os.path.join(model_download_root, file_name)
        else:
            raise RuntimeError(f"Score {name} not found; available scores = {available_scores()}")
        print('load checkpoint from %s'%model_path)
        
    if name == "BLIP":
        state_dict = torch.load(model_path, map_location='cpu')
        med_config = os.path.expanduser("~/.cache/metric_models/med_config.json")
        model = BLIPScore(med_config=med_config, device=device).to(device)
        model.blip.load_state_dict(state_dict['model'],strict=False)
    elif name == "CLIP":
        model = CLIPScore(download_root=model_download_root, device=device).to(device)
    elif name == "Aesthetic":
        state_dict = torch.load(model_path, map_location='cpu')
        model = AestheticScore(download_root=model_download_root, device=device).to(device)
        model.mlp.load_state_dict(state_dict,strict=False)
    elif name == "ImageReward":
        model_path = os.path.expanduser("~/.cache/metric_models/ImageReward.pt")
        state_dict = torch.load(model_path, map_location='cpu')
        med_config = os.path.expanduser("~/.cache/metric_models/med_config.json")
        model = ImageReward(device=device, med_config=med_config).to(device)
        model.load_state_dict(state_dict,strict=False)
    elif name == "HPS":
        # params = torch.load(_SCORES[name])['state_dict']
        model = HPS(download_root=model_download_root, device=device).to(device)
    elif name == "PickScore":
        model = PickScore(device = device)

    else:
        raise RuntimeError(f"Score {name} not found; available scores = {available_scores()}")
    
    print("checkpoint loaded")
    model.eval()

    return model