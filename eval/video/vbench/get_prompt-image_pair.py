import os
import pandas as pd
import json
import random

dimensions = [
    "subject_consistency",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "dynamic_degree",
]

def sample_images_to_json(image_path: str, json_path: str, num_samples: int, dimension: str, seed: int = 42):
    
    suffix=".jpg"
    all_images = [f[:-3] for f in os.listdir(image_path) if f.lower().endswith(suffix)]

    # sample
    if seed is not None:
        random.seed(seed)
    num_samples = min(num_samples, len(all_images))
    sampled = random.sample(all_images, num_samples)

    json_data = [
        {
            "prompt_en": p,
            "dimension": dimension,
            "image_path": os.path.join(image_path, p+suffix[1:])
        }
        for p in sampled
    ]

    # save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"save {num_samples} prompt-image pairs to {json_path}")


if __name__ == "__main__":
    num_samples = 600
    image_path = "./vbench/origin_crop"
    json_file = f"./vbench/sampled_images_{num_samples:02d}.json"
    sample_images_to_json(image_path, json_file, num_samples=num_samples, dimension=dimensions)
