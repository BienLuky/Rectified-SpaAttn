import argparse
import json
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench_path", type=str)
    parser.add_argument("--vr_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    if args.vbench_path:
        folder = os.path.join(base_dir, args.vbench_path)
        folder_path = Path(folder)

        vbench_scores = {}
        for file in folder_path.glob("*_eval_results.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            dim = next(iter(data))
            value = data[dim][0]
            vbench_scores[dim] = value

        for dim, value in vbench_scores.items():
            print(f"{dim}: {value:.4f}")

    if args.vr_path:
        json_path = os.path.join(base_dir, args.vr_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dim = next(iter(data))
        value = data[dim] 
        print(f"{dim}: {value:.4f}")
        