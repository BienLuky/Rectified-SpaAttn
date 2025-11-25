#!/bin/bash

# Define paths and arguments
export HF_ENDPOINT=https://hf-mirror.com
#metric="ImageReward"

fp_path="$YOUR_PATH/project/attn_quant/diffuser-dev/examples/cogvideo_attn/diffuser-dev-220flux/examples/flux/logs/calib_data/coco/generated_images_30"
prompt_path="/mnt/nas/lizhikai/Liu/GRAT/eval/image/coco_1024.txt"

path_file="paths.txt"

# 读取文件中的路径并遍历
while read -r path; do
    #echo "当前路径: $path"
	img_dir=$path
	log_file="${img_dir}/../eval.txt"
	echo $img_dir

	python fid_score.py \
			--path "$fp_path" "$img_dir" \
			--log_file "$log_file"

	python test_score.py \
		--prompts_path $prompt_path \
		--metric "CLIP" \
		--img_dir $img_dir \
		--log_file $log_file

	python test_score.py \
		--prompts_path $prompt_path \
		--metric "ImageReward" \
		--img_dir $img_dir \
		--log_file $log_file

	python eval_image_diff.py \
		   --path1 $fp_path  \
		--path2 $img_dir \


done < "$path_file"



