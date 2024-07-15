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
  
#semantic need another parameter
dir_path="models_file_path"
  file_names=("$dir_path"/*)
  for file_name in "${file_names[@]}"; do
      python cc1m_instance_segmentation.py --model_name ${file_name##*/} --model_path "$file_name"/*.pth --output ./output --clean_path "" --adv_path ""
done