#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --job-name=job_sa_cont_2
#SBATCH --output=class_sa_cont_2.out
#SBATCH --error=error_sa_cont_2.out
#SBATCH --account=rrg-mageed
#SBATCH --mail-user=rrs99@cs.ubc.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
module load cuda cudnn
source /home/rrs99/projects/rrg-mageed/rrs99/venvs/check/bin/activate

export GLOVE_PATH=/home/rrs99/projects/rrg-mageed/rrs99/code/sd/glove_embedding/glove.6B.200d.txt
export MODEL_INFO=context_2
export TRAIN_SET=/home/rrs99/projects/rrg-mageed/rrs99/code/Speech_Act_Classifier/data/switchboard_train/train_context_left_2.json
export DEV_SET=/home/rrs99/projects/rrg-mageed/rrs99/code/Speech_Act_Classifier/data/switchboard_dev/dev_context_left_2.json
export TEST_SET=/home/rrs99/projects/rrg-mageed/rrs99/code/Speech_Act_Classifier/data/switchboard_test/test_context_left_2.json

python3 train_sa.py \
        --glove_path=$GLOVE_PATH \
        --model_info=$MODEL_INFO \
        --num_epochs=30 \
        --train_set=$TRAIN_SET \
        --dev_set=$DEV_SET \
        --test_set=$TEST_SET \
        --num_sentences=3 \
        --left_context_only=True \
