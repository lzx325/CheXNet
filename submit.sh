job_name="Apr18-weighted1-noaug-lr_1e-4"
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=1-00:00:00 # DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --output tmp/${job_name}.txt
#SBATCH --error tmp/${job_name}.err.txt
srun --ntasks=1 python -u chexnet_test.py ./${job_name}/best_model.pkl
EOF

job_name="Apr18-weighted2-noaug-lr_1e-4"
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=1-00:00:00 # DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --output tmp/${job_name}.txt
#SBATCH --error tmp/${job_name}.err.txt
srun --ntasks=1 python -u chexnet_test.py ./${job_name}/best_model.pkl
EOF

job_name="Apr18-unweighted-aug-lr_1e-4"
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=1-00:00:00 # DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --output tmp/${job_name}.txt
#SBATCH --error tmp/${job_name}.err.txt
srun --ntasks=1 python -u chexnet_test.py ./${job_name}/best_model.pkl
EOF
