

# #####   Inference   #####
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rectified
# echo "The environment rectified has been activated."

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 
# python ./experiments/multigpu_flux.py --image_path ../samples/flux_ours_step50



#####   Inference   #####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grat
echo "The environment grat has been activated."

export CUDA_VISIBLE_DEVICES=0,1
CUDA_LAUNCH_BLOCKING=1 python ./evaluation/experiments/multigpu_flux.py --image_path ../samples/flux_ours_step50 --enable_teacache
