#!/bin/bash -l

#SBATCH --job-name=LCA_train
#SBATCH -N 2
#SBATCH --constraint=gpu_count:4
#SBATCH -p power9 
#SBATCH --qos=long 
#SBATCH --time=02-00:00:00
#SBATCH --output=train_lca.txt
#SBATCH --signal=USR2@360
 
# To submit your job: `sbatch <script.sh>
# To view your job ID and status: `squeue -u <username>
# To cancel your job: `scancel <job_id>

# the following allows an executable to run directly as an openmpi singleton
#export OMPI_MCA_pmix=pmix112 


executable=/home/mteti/OpenPV/build/tests/BasicSystemTest/Release/BasicSystemTest


module purge
module avail
module load gcc/8.3.0 cuda/10.1 openmpi/p9/4.0.2-gcc_8.3.0
module list

#echo $PATH
#echo $LD_LIBRARY_PATH
#echo $MODULEPATH

lua learn_imagenet_dict.lua > learn_imagenet_dict.param


# -N * number of nodes = np; batchwidth = np >= n_batch / # GPUs
mpirun \
    -np 16 \
    -N 8 \
    --bind-to none \
    --oversubscribe \
    $executable \
        -t \
        -batchwidth 16 \
        -rows 1 \
	-columns 1 \
	-p learn_imagenet_dict.param


## to run this script dependent on a previous run with <jobid> (from squeue) finishing:
## sbatch --dependency=afterany:<jobid> ~/FaceForensics/scripts/faceforensicsbatch.sh
