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
NUM_MONTE_CARLO=50


# python ../bayesian_neural_network.py --data_dir=${DATA_DIR} \
#     --model_dir=${MODEL_DIR} --num_monte_carlo=${NUM_MONTE_CARLO} >bnn.log

lr_array=(0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3)

for lr in ${lr_array[@]}
do : 
python ../bayesian_neural_network.py --data_dir=/vol/home/yaojian/zhoudoao/projects/datasets/mnist \
    --model_dir=/vol/home/yaojian/zhoudoao/projects/model/bnn \
    --num_epochs=500 \
    --viz_steps=400\
    --num_monte_carlo=20 \
    --batch_size=128 \
    --learning_rate=$lr \
    --fake_data=False > "bnn_lr_${lr}.log"
done