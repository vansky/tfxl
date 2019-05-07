#!/bin/bash
#SBATCH --partition=gpuk80
#SBATCH --exclude=gpu019
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --job-name=TFeval
#SBATCH --mail-type=end
#SBATCH --mail-user=EMAIL
#SBATCH --output=eval_wt103.out
#SBATCH --error=eval_wt103.err
#SBATCH --time=0-2:00:0

module load python/3.6-anaconda
source activate pytorch-1.0.0

module load cuda/8.0

corpussize='200k';
nhid=200;
seed=$RANDOM;

time python eval.py \
        --cuda \
        --tgt_len 150 \
        --mem_len 0 \
	--data eval_templates/ \
	--dataset wt103 \
        --batch_size 1 \
	--trainfname all_test_sents.txt \
	--validfname all_test_sents.txt \
	--testfname all_test_sents.txt \
	--split test \
	--vocab_file \
	--work_dir MODEL_SUPER_DIR/MODEL_SUB_DIR/
