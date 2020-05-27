#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -J tensorflow_job
#SBATCH --partition=128G
#SBATCH --ntasks=1

source /vol/home/yaojian/zhoudoao/run_me_first_bashrc
conda init
conda activate zda_tfp_env

DATA_DIR=/vol/home/yaojian/zhoudoao/projects/datasets/mnist
MODEL_DIR=/vol/home/yaojian/zhoudoao/projects/model/bnn


# Todo: python bnn.py   -2020.5.26
