#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J universeg            # Job name
#SBATCH -o universeg_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e universeg_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 01:30:00         # Run time (hh:mm:ss) - 10 min
#SBATCH --mem-per-cpu=3G    # Memory per core demandes (96 GB = 3GB * 32 cores)

module load cesga/system miniconda3/22.11.1-1
conda activate TFM_final

# Para asegurarse de que se carga correctamente el entorno
conda deactivate
conda activate TFM_final
# cd /mnt/netapp2/Store_uni/home/usc/ci/fgs/git_repo/Codigo_TFG
srun python combined_pipeline_universeg.py combined_config_universeg.yaml
# srun python see_med_sam_architecture.py
echo "done"                 # Write this message on the output file when finished