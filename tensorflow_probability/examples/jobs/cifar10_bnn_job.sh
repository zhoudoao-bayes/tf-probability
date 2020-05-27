#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -J tensorflow_job
#SBATCH --partition=128G
#SBATCH --ntasks=2

source /vol/home/yaojian/zhoudoao/run_me_first_bashrc

conda init
conda activate zda_tfp_env

DATA_DIR=/vol/home/yaojian/zhoudoao/projects/datasets/cifar10/
MODEL_DIR=/vol/home/yaojian/zhoudoao/projects/model/cifar10_bnn
NUM_MONTE_CARLO=50
python ../cifar10_bnn.py --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} --num_monte_carlo=${NUM_MONTE_CARLO} >cifar10_bnn.log