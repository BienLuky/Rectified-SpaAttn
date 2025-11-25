
#####   Evaluation   #####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrics
echo "The environment metrics has been activated."

fp_path="Your_Path/image/evaluation/samples/flux_ours_step50"
prompt_path="Your_Path/image/coco_1024.txt"
img_dir="Your_Path/image/evaluation/samples/flux_ours_step50"


log_file="Your_Path/image/evaluation/samples/eval.txt"
echo $img_dir

python ./evaluation/fid_score.py --path "$fp_path" "$img_dir" --log_file "$log_file"
python ./evaluation/test_score.py --prompts_path $prompt_path --metric "CLIP" --img_dir $img_dir --log_file $log_file
python ./evaluation/test_score.py --prompts_path $prompt_path --metric "ImageReward" --img_dir $img_dir --log_file $log_file
python ./evaluation/eval_image_diff.py --path1 $fp_path  --path2 $img_dir --log_file $log_file