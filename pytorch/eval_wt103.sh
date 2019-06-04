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
#SBATCH --array=400-404,410-414,420-424,430-434,440-444,450-454,460-464,470-474,480-484
#80-84
module load python/3.6-anaconda
source activate pytorch-1.0.0
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

workdir="${dir_prefix}-${nlayer}-${corpussize}-${nhid}-${Lflag}"
subdir=$(find ./cont-${workdir}-wt103/* -maxdepth 1 -type d -name '[^.]?*' -printf %f -quit)
#subsubdir=$(find ./${workdir}-wt103/${subdir}/* -maxdepth 1 -type d -name '[^.]?*' -printf %f -quit)

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
	--work_dir cont-${workdir}-wt103/${subdir}/ > eval_output/${workdir}.output
	#--work_dir ${workdir}-wt103/${subdir}/${subsubdir}/ > eval_output/${workdir}.output
