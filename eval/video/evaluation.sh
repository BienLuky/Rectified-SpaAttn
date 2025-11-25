

export CUDA_VISIBLE_DEVICES=0

SAMPLE_PATH="hunyuan_ours_step50"
SCORE_PATH="score_ours_step50"

#####   Evalution   #####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_eval
echo "The environment video_eval has been activated."

# Vbench Scores
python ./vbench/run_vbench.py --video_path ./samples/${SAMPLE_PATH} --save_path ./samples/${SCORE_PATH}

# VR Scores
cd ./VisionReward/
python inference-video.py --score --path ./samples/${SAMPLE_PATH}
cd ..

# Print Scores
python ./vbench/print_scores.py --vbench_path ../samples/${SCORE_PATH} --vr_path ../VisionReward/${SAMPLE_PATH}.json
