#!/bin/bash
#SBATCH --partition=gpuk80
#SBATCH --exclude=gpu019
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=TFanalyze
#echo SBATCH --time=0-4:00:00 #cpu
#SBATCH --time=0-1:00:00 #gpu
#SBATCH --mail-type=end
#SBATCH --mail-user=vansky@jhu.edu
#SBATCH --output=logs/analyze_wt103.%a.out
#SBATCH --error=logs/analyze_wt103.%a.err
#echo SBATCH --array=0-4,10-14,20-24,30-34,40-44,50-54,60-64,70-74,80-84
#SBATCH --array=400-404,410-414,420-424,430-434,440-444,451-454,460-462,470-474,480-484

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
subdir=$(find ./${workdir}-wt103/* -maxdepth 1 -type d -name '[^.]?*' -printf %f -quit)
#subsubdir=$(find ./${workdir}-wt103/${subdir}/* -maxdepth 1 -type d -name '[^.]?*' -printf %f -quit)

python filter_tf_output.py eval_output/${workdir}.output > eval_output/${workdir}-filt.output
python ../../LM_syneval/src/LM_eval-score.py --model_type tf --output_file eval_output/${workdir}-filt.output
python ../../LM_syneval/src/analyze_results.py --results_file eval_output/${workdir}-filt_results.pickle --model_type rnn --mode overall --out_dir results_${workdir}
