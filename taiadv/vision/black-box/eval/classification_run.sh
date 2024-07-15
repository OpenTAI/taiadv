#!/bin/bash
  # Parameters
  #SBATCH --cpus-per-task=4
  #SBATCH --gres=gpu:1
  #SBATCH --error=/vhome/quzhijiu/%j/%j_0_log.err
  #SBATCH --gpus-per-node=8
  #SBATCH --job-name=deit
  #SBATCH --mem=320GB
  #SBATCH --nodes=1
  #SBATCH --ntasks-per-node=8
  #SBATCH --open-mode=append
  #SBATCH --output=/vhome/quzhijiu/%j/%j_0_log.out
  #SBATCH --partition=fvl
  #SBATCH --qos=high
  #SBATCH --signal=USR1@120
  #SBATCH --time=2800
  #SBATCH --wckey=submitit
  # command

python generate.py --model_name "" --model_path "" --data_path ""
python cc1m_classification.py --model_name "" --model_path "" --output "" --data_path ""