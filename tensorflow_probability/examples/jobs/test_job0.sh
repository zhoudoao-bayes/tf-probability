#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -J tensorflow_job
#SBATCH --partition=128G
#SBATCH --ntasks=2

source /vol/home/yaojian/zhoudoao/run_me_first_bashrc
conda init
conda activate zda_tfp_env

python -V
python ../test_gpu.py >> test_gpu.log
