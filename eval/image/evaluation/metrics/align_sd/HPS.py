import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
import os

class HPS(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False, 
                                                     download_root=download_root)

        params = torch.load(download_root+'/hpc.pt')['state_dict']
        self.model.load_state_dict(params)
        
        # if device == "cpu":
        #     self.clip_model.float()
        # else:
            # clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.model.logit_scale.requires_grad_(False)


    def score(self, prompt, image_path):

        if (os.path.isdir(image_path)):
            indices, rewards = self.inference_rank(prompt, image_path)
            return indices, rewards
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        image = image.to(self.device)
        text = clip.tokenize(prompt, context_length=77, truncate=True).to(self.device)

        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        hps = image_features @ text_features.T
        hps = hps.diagonal()
        
        return None, hps.squeeze().detach().cpu().numpy().item()


    def inference_rank(self, prompt, generations_list):
        
        text = clip.tokenize(prompt, context_length=77, truncate=True).to(self.device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        txt_set = []
        img_set = []
        file_list = os.listdir(generations_list)
        file_list.sort(key = lambda x: int(x.split(".")[0]))
        for i, img_name in enumerate(file_list):
            # image encode
            img_path = os.path.join(generations_list, img_name)
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            img_set.append(image_features)
            txt_set.append(text_features[i].unsqueeze(0))
            
        txt_features = torch.stack(txt_set, 0).float() # [image_num, feature_dim]
        img_features = torch.stack(img_set, 0).float() # [image_num, feature_dim]
        rewards = torch.bmm(txt_features, img_features.transpose(2,1)).squeeze()
        # hps = image_features @ text_features.T
        # hps = hps.diagonal()

        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()