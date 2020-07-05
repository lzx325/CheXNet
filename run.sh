#!/usr/bin/env bash
if [ -z "$1" ] || [ "$1" = "train" ]; then
    srun --ntasks=1 --pty python -u chexnet_train_aug_epoch.py || { echo "command exited with status $?"; exit 2; }
elif [ "$1" = "test" ]; then 
    srun --ntasks=1 --pty python -u chexnet_test.py || { echo "command exited with status $?"; exit 2; }
elif [ "$1" = "exp" ]; then
    srun --ntasks=1 --pty python -u data_parallel_tutorial.py || { echo "command exited with status $?"; exit 2; }
elif [ "$1" = "batch" ]; then
    srun --ntasks=1 --pty python -u chexnet_train_refined.py ./test3 weighted1 False || { echo "command exited with status $?"; exit 2; }
else
    echo "Illegal Argument" 1>&2; exit 1;
fi