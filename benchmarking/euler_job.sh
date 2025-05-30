#!/bin/bash

module load stack/2024-06  gcc/12.2.0 python_cuda/3.11.6 eth_proxy

for i in {400..400}
do
   for epistemic_idx in {0..2499}
   do
      echo "Solving with random vector of epistemic_idx $epistemic_idx for i $i"
      sbatch --mem-per-cpu=2G --gpus=1 --gres=gpumem:24g --output "/cluster/scratch/manishp/4441_log/slurm_$epistemic_idx.log" --wrap="python sampling-gpmpc/benchmarking/simulate_forward_sampling_car.py -epistemic_idx $epistemic_idx"
   done
done