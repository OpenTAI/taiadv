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

file_name="/vhome/quzhijiu/lab_lab/objectdetection/models/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco"
python general_eval.py "$file_name"/*.py "$file_name"/*.pth
