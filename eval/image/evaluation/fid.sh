#!/bin/bash

fp_path="$YOUR_PATH/diffuser-dev/examples/Flux/logs/final/BASIC8/generated_images_30"
base_dir="$YOUR_PATH/diffuser-dev/examples/Flux/logs/final/Perrow_QKsmooth8_PV8/generated_images_30"
log_file="./logs/test.txt"

python fid_score.py \
        --path "$fp_path" "$base_dir" \
        --log_file "$log_file"
# Full Suite PTQ FID eval
# for dir in "$base_dir"/*; do
#     if [ -d "$dir" ]; then
#         echo "FID evaluation: QUANT: $dir FP: $fp_path"
#         python evaluation/fid_score.py \
#         --path "$fp_path" "$dir" \
#         --log_file "$log_file"
#     fi
# done


