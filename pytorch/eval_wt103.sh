#!/bin/bash
#SBATCH --partition=gpuk80
#SBATCH --exclude=gpu019
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=TFeval
#echo SBATCH --time=0-4:00:0 #cpu
#SBATCH --time=0-0:30:0 #gpu
#SBATCH --mail-type=end
#SBATCH --mail-user=vansky@jhu.edu
#SBATCH --output=logs/eval_wt103.%a.out
#SBATCH --error=logs/eval_wt103.%a.err
#SBATCH --array=0-4,10-14,20-24,30-34,40-44,50-54,60-64,70-74,80-84

module load python/3.6-anaconda
source activate pytorch-1.0.0

#EXCLUDE: git python
module load cuda/8.0

#Lflag=$SLURM_ARRAY_TASK_ID

#seed=$(($Lflag % 10));
#corpussize='200k';
#nhid=200;
#seed=$RANDOM;

Lflag=$SLURM_ARRAY_TASK_ID

nlayer=$(($Lflag % 1000 / 100 + 2));

if [[ $(($Lflag / 1000)) -eq 0 ]]; then
    attn_type=2;
    mem_len=0;
elif [[ $(($Lflag / 1000)) -eq 1 ]]; then
    attn_type=0;
    mem_len=150;
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

workdir="LMv-${nlayer}-${corpussize}-${nhid}-${Lflag}"
subdir=$(find ./${workdir}-wt103/*-wt103 -maxdepth 1 -type d -name '[^.]?*' -printf %f -quit)
subsubdir=$(find ./${workdir}-wt103/${subdir}/* -maxdepth 1 -type d -name '[^.]?*' -printf %f -quit)

time python eval.py \
        --cuda \
        --tgt_len 150 \
        --mem_len ${mem_len} \
	--data ../../LM_syneval/my_templates/ \
	--vocab_file glove.num_unk.vocab \
	--dataset wt103 \
        --batch_size 1 \
	--trainfname all_test_sents.txt \
	--validfname all_test_sents.txt \
	--testfname all_test_sents.txt \
	--split test \
	--work_dir ${workdir}-wt103/${subdir}/${subsubdir}/ > eval_output/${workdir}.output

#        --cuda \
#--clamp_len 400 \
#if [[ $1 == 'train' ]]; then
#	--same_length \
#        --batch_size 60 \
#        --gpu0_bsz 4
#        --tgt_len 150 \
#        --mem_len 150 \
#        --eval_tgt_len 150 \
#        ${@:2}

#time python eval.py  --dataset wt103 --tgt_len 1 --mem_len 150 --clamp_len 400 --same_length --split test --trainfname wiki.train.tokens --validfname wiki.valid.tokens --testfname small.test.tokens --work_dir 
