#!/bin/bash
#SBATCH --partition=gpuk80
#SBATCH --exclude=gpu019
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=TFXLT
#SBATCH --mail-type=end
#SBATCH --mail-user=EMAIL
#SBATCH --output=train_wt103.out
#SBATCH --error=train_wt103.err
#SBATCH --time=0-20:0:0

module load python/3.6-anaconda
source activate pytorch-1.0.0

#EXCLUDE: git python
module load cuda/8.0

corpussize='200k';
nhid=200;
seed=$RANDOM;

python train.py \
        --work_dir WORKING_DIR \
        --cuda \
        --tgt_len 150 \
        --mem_len 0 \
        --eval_tgt_len 150 \
        --batch_size 20 \
	--trainfname wiki.train.tokens \
	--validfname wiki.valid.tokens \
	--testfname wiki.test.tokens \
	--vocab_file grnn.vocab \
        --gpu0_bsz 4 \
	--attn_type 2 \
	--seed ${seed} \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --n_layer 4 \
        --d_model 400 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.2 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000
