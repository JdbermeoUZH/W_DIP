#!/bin/bash
#SBATCH  --output=batch_jobs_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/minga/jbermeo/conda/etc/profile.d/conda.sh
conda activate BMIC_project
cd /scratch_net/minga/jbermeo/W_DIP
cat sbatch_test.sh

python -u WDIP.py --data_path datasets/micro_good/blur/ --save_path results/micro_good --dataset_name micro_good --ksize_path kernel_estimates/micro_good_kernel.yaml --save_frequency 10000
