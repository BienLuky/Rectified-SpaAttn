import torch 
import numpy as np
import os
import shutil
import json

def prepare_coco_text_and_image(json_file, dataset_path):
    info = json.load(open(json_file, 'r'))
    annotation_list = info["annotations"]
    image_caption_dict = {}
    for annotation_dict in annotation_list:
        if annotation_dict["image_id"] in image_caption_dict.keys():
            image_caption_dict[annotation_dict["image_id"]].append(annotation_dict["caption"])
        else:
            image_caption_dict[annotation_dict["image_id"]] = [annotation_dict["caption"]]
    captions = list(image_caption_dict.values())
    image_ids = list(image_caption_dict.keys())
    
    active_captions = []
    for texts in captions:
        active_captions.append(texts[0])
        
    image_paths = []
    for image_id in image_ids:
        image_paths.append(dataset_path+f"COCO_val2014_{image_id:012}.jpg")
    return active_captions, image_paths