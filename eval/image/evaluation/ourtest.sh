#!/bin/bash
# =================================================================================
# /mnt/nas/lizhikai/Liu/GRAT/eval/image/evaluation/metrics/utils.py /home/lizhikai/.cache/metrics_models

# =================================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrics
echo "The environment metrics has been activated."

fp_path="/mnt/nas/lizhikai/Liu/GRAT/eval/image/evaluation/samples/flux_base_50_step50"
prompt_path="/mnt/nas/lizhikai/Liu/GRAT/eval/image/coco_1024_sub.txt"
img_dir="/mnt/nas/lizhikai/Liu/GRAT/eval/image/evaluation/samples/flux_base_50_step50"

#echo "当前路径: $path"

log_file="/mnt/nas/lizhikai/Liu/GRAT/eval/image/eval.txt"
echo $img_dir

python fid_score.py --path "$fp_path" "$img_dir" --log_file "$log_file"

python test_score.py --prompts_path $prompt_path --metric "CLIP" --img_dir $img_dir --log_file $log_file

python test_score.py --prompts_path $prompt_path --metric "ImageReward" --img_dir $img_dir --log_file $log_file

python eval_image_diff.py --path1 $fp_path  --path2 $img_dir





