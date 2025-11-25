

export CUDA_VISIBLE_DEVICES=0,1,2,3


#####   HunyuanVideo   #####
SAMPLE_PATH="hunyuan_ours_step50"
SCORE_PATH="score_ours_step50"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rectified
echo "The environment rectified has been activated."
python ./experiments/multigpu_hunyuan.py --video_path ../samples/${SAMPLE_PATH}

# #####   Wan21 T2V   #####
# SAMPLE_PATH="wan21t2v_ours_step50"
# SCORE_PATH="score_ours_step50"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rectified
# echo "The environment rectified has been activated."
# python ./experiments/multigpu_wan21t2v.py --video_path ../samples/${SAMPLE_PATH}


# #####   Wan21 I2V   #####
# SAMPLE_PATH="wan21t2v_ours_step50"
# SCORE_PATH="score_ours_step50"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rectified
# echo "The environment rectified has been activated."
# python ./experiments/multigpu_wan21i2v.py --video_path ../samples/${SAMPLE_PATH}


