# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import torch.nn as nn
import os



class PickScore(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        processor_name_or_path = "$YOUR_PATH/.cache/metric_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23"
        model_pretrained_name_or_path = "$YOUR_PATH/.cache/metric_models/models--yuvalkirstain--PickScore_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
        
        # if device == "cpu":
        #     self.clip_model.float()
        # else:
        #     clip.model.convert_weights(self.clip_model) # Actually this line is unnecessary since clip by default already on float16

        # # have clip.logit_scale require no grad.
        # self.clip_model.logit_scale.requires_grad_(False)


    def score(self, prompt, image_path):
        if (os.path.isdir(image_path)):
            indices, rewards = self.inference_rank(prompt, image_path)
            return indices, rewards
        
        image_inputs = self.processor(
        images=Image.open(image_path),
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
        ).to(self.device)
    
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            # probs = torch.softmax(scores, dim=-1)
        
        return None, scores.detach().cpu().numpy().item()

    def inference_rank(self, prompt, generations_list):

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            txt_set = []
            img_set = []
            file_list = os.listdir(generations_list)
            file_list.sort(key = lambda x: int(x.split(".")[0]))
            for i, img_name in enumerate(file_list):
                # image encode
                img_path = os.path.join(generations_list, img_name)
                image_inputs = self.processor(
                    images=Image.open(img_path),
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                    ).to(self.device)
                
                image_embs = self.model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                img_set.append(image_embs)
                txt_set.append(text_embs[i].unsqueeze(0))

            txt_features = torch.stack(txt_set, 0).float() # [image_num, feature_dim]
            img_features = torch.stack(img_set, 0).float() # [image_num, feature_dim]

            # batch
            rewards = self.model.logit_scale.exp() * (torch.bmm(txt_features, img_features.transpose(2,1)).squeeze())
            
            # get probabilities if you have multiple images to choose from
            # rewards = torch.softmax(rewards, dim=-1)

            _, rank = torch.sort(rewards, dim=0, descending=True)
            _, indices = torch.sort(rank, dim=0)
            indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()