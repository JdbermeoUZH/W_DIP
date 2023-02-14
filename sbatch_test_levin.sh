#!/bin/bash
#SBATCH  --output=batch_jobs_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/minga/jbermeo/conda/etc/profile.d/conda.sh
conda activate BMIC_project
cd /scratch_net/minga/jbermeo/W_DIP
cat sbatch_test.sh

python -u WDIP.py --save_frequency 10000
