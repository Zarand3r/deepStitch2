#!/bin/bash
BATCH --time=30:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH --mail-user=fluongo@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#Submit this script with: sbatch thefilename

#Activate conda env
conda activate tensor1
nvidia-smi
echo "Environment activated, running file...."

cd ../
cd preprocessing
cd optical_flow

sbatch generate_flows
#python lightning_train.py --datadir /central/groups/tensorlab/rbao/usc_data/classification_data --include_classes "01_02_03_07_13"

