#!/bin/bash
#SBATCH --partition=gpuk80
#SBATCH --exclude=gpu019
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=TFXLT
#SBATCH --time=0-20:0:0
#SBATCH --mail-type=end
#SBATCH --mail-user=vansky@jhu.edu
#SBATCH --output=logs/train_wt103.%a.out
#SBATCH --error=logs/train_wt103.%a.err
#SBATCH --array=473 #444,461,462,464,480,481
#1000-1004,1010-1014,1020-1024,1030-1034,1040-1044,1050-1054,1060-1064,1070-1074,1080-1084
#echo SBATCH --array=1400-1404,1410-1414,1420-1424,1430-1434,1440-1444,1450-1454,1460-1464,1470-1474,1480-1484

module load python/3.6-anaconda
source activate pytorch-1.0.0

#EXCLUDE: git python
module load cuda/8.0

Lflag=$SLURM_ARRAY_TASK_ID

nlayer=$(($Lflag % 1000 / 100 + 2));

if [[ $(($Lflag / 1000)) -eq 0 ]]; then
    attn_type=2;
    mem_len=0;
    dir_prefix='LMv';
elif [[ $(($Lflag / 1000)) -eq 1 ]]; then
    attn_type=0;
    mem_len=150;
    dir_prefix='LMvxl';
fi

if [[ $(($Lflag % 100 / 10)) -eq 0 ]]; then
    corpussize='20';
elif [[ $(($Lflag % 100 / 10)) -eq 1 ]]; then
    corpussize='10';
elif [[ $(($Lflag % 100 / 10)) -eq 2 ]]; then
    corpussize='40';
elif [[ $(($Lflag % 100 / 10)) -eq 3 ]]; then
    corpussize='20';
elif [[ $(($Lflag % 100 / 10)) -eq 4 ]]; then
    corpussize='10';
elif [[ $(($Lflag % 100 / 10)) -eq 5 ]]; then
    corpussize='40';
elif [[ $(($Lflag % 100 / 10)) -eq 6 ]]; then
    corpussize='20';
elif [[ $(($Lflag % 100 / 10)) -eq 7 ]]; then
    corpussize='10';
elif [[ $(($Lflag % 100 / 10)) -eq 8 ]]; then
    corpussize='40';
fi

if [[ $(($Lflag % 10)) -eq 0 ]]; then
    nhid='100';
elif [[ $(($Lflag % 10)) -eq 1 ]]; then
    nhid='200';
elif [[ $(($Lflag % 10)) -eq 2 ]]; then
    nhid='400';
elif [[ $(($Lflag % 10)) -eq 3 ]]; then
    nhid='800';
elif [[ $(($Lflag % 10)) -eq 4 ]]; then
    nhid='1600';
fi

#seed=$(($Lflag % 10));
seed=$RANDOM;
#seed=$LFlag;

# Train
##        --data ../data/rnn_tf_cmp/ \
##	--trainfname wiki103_${corpussize}_${corpusvar}.train \
##	--validfname wiki103.valid \
##	--testfname sample.test \
##        --dataset custom \
#        --n_layer 2 \
#        --d_model ${nhid} \
#        --n_head 4 \
#        --d_head 16 \
#        --d_inner 100 \
#        --dropout 0.2 \
#        --dropatt 0.0 \
#        --optim adam \
#        --lr 1 \
#        --warmup_step 10000 \

time python train.py \
        --cuda \
        --tgt_len 150 \
        --mem_len ${mem_len} \
        --eval_tgt_len 150 \
        --batch_size 20 \
	--trainfname wiki.train.tokens \
	--validfname wiki.valid.tokens \
	--testfname wiki.test.tokens \
	--vocab_file glove.num_unk.vocab \
        --gpu0_bsz 4 \
	--attn_type ${attn_type} \
	--seed ${seed} \
	--work_dir ${dir_prefix}-${nlayer}-${corpussize}-${nhid}-${Lflag} \
        --data ../data/wikitext-103-${corpussize}/ \
        --dataset wt103 \
        --n_layer ${nlayer} \
        --d_model ${nhid} \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.2 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000
