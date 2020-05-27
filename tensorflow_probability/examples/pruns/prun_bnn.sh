# Remote debug in interactive mode
# srun -n 2 --partition=128G --pty /bin/bash

# global setting
source ~/zhoudoao/run_me_first_bashrc
conda activate zda_tfp_env

# local setting
DATA_DIR=/vol/home/yaojian/zhoudoao/projects/datasets/mnist
MODEL_DIR=/vol/home/yaojian/zhoudoao/projects/model/bnn

lr_array=(0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3)

for lr in ${lr_array[@]}
do :
    echo 'learning rate: ' $lr
    prun python ../bayesian_neural_network.py --data_dir=/vol/home/yaojian/zhoudoao/projects/datasets/mnist \
    --model_dir=/vol/home/yaojian/zhoudoao/projects/model/bnn \
    --viz_steps=400 \
    --num_monte_carlo=20 \
    --num_epochs=2 \
    --batch_size=128 \
    --learning_rate=$lr \
    --fake_data=False
done