import pandas as pd
import json

dimensions = [
    "subject_consistency",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "dynamic_degree",
]

def sample_prompts_to_json(csv_path: str, json_path: str, num_samples: int, dimension: str, seed: int = 42):
    """
    Randomly select a specified number of prompts from a CSV file and save them 
    to a JSON file in the given format.

    Args:
        csv_path (str): Path to the input CSV file.
        json_path (str): Path to the output JSON file.
        num_samples (int): Number of samples to randomly select.
        dimension (str): Dimension label shared by all prompts.
        seed (int): Random seed for reproducibility (default: 42).
    """

    # load CSV
    df = pd.read_csv(csv_path)

    # samples
    num_samples = min(num_samples, len(df))
    sampled = df["prompt"].sample(n=num_samples, random_state=seed).tolist()

    json_data = [
        {
            "prompt_en": p,
            "dimension": dimension
        }
        for p in sampled
    ]

    # save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"save {num_samples} prompts to {json_path}")


if __name__ == "__main__":
    num_samples = 600
    csv_file = "./vbench/PenguinVideoBenchmark.csv"
    json_file = f"./vbench/sampled_prompts_{num_samples:02d}.json"
    sample_prompts_to_json(csv_file, json_file, num_samples=num_samples, dimension=dimensions)
