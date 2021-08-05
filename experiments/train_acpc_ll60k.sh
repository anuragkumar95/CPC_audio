#!/bin/bash

# Script for the LIS slurm cluster

set -x

RVERB=""  # =-v

REMOTE_USER=ricard.marxer
REMOTE_HOST=cluster_lis

# location of the main repository (contains data/)
CPC_DIR="$( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))"
REMOTE_CPC_DIR=/home/ricard.marxer/dev/CPC_audio
REMOTE_MINICONDA_DIR=/home/ricard.marxer/miniconda3
REMOTE_DATA_DIR=/home/ricard.marxer/data/LibriLight_segmented/unlab-60k
REMOTE_DATA_EXT=".flac"
REMOTE_DATA_SPLITS=/home/ricard.marxer/data/LibriLight_segmented/splits/unlab-60k
REMOTE_LIBRISPEECH_DIR=/home/ricard.marxer/data/LibriSpeech_wav
REMOTE_LIBRISPEECH100_SPLITS=/home/ricard.marxer/scratch/LibriSpeech100_labels_split

# top-level directory for experiments
REMOTE_EXPERIMENT_RUNDIR=/home/ricard.marxer/runs/

# adjust the main loop
# (it can go over .yaml files, over hyperparameters, etc.

#"--CPCCTCNumMatched 12 --nPredicts 8 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 6 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 4 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 10 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 8 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 6 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 4 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 10 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 12" \

#"--CPCCTCNumMatched 12 --nPredicts 2 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 12 --nPredicts 2 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 12" \
#"--CPCCTCNumMatched 24 --nPredicts 12 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 24" \
#"--CPCCTCNumMatched 24 --nPredicts 12 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 24" \
#"--CPCCTCNumMatched 24 --nPredicts 14 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 24" \
#"--CPCCTCNumMatched 24 --nPredicts 14 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 24" \
#"--CPCCTCNumMatched 24 --nPredicts 16 --CPCCTCSkipBeg 0 --CPCCTCSkipEnd 24" \
#"--CPCCTCNumMatched 24 --nPredicts 16 --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 24" \

for PARAMS in \
"--CPCCTCNumMatched 12 --nPredicts 8" \
; do

# low-level directory for experiments
EXP_TAG=remote_cpcctc_60k_mp
PRINT_PARAMS=$(echo $PARAMS | tr -d ' ' | sed -e 's/-\+/_/g')
NAME=test_cpcctc${PRINT_PARAMS}
DIR=$EXP_TAG/$NAME
EXP_DIR=$REMOTE_EXPERIMENT_RUNDIR/$DIR

echo $EXP_DIR

ssh -q $REMOTE_USER@$REMOTE_HOST mkdir -p $EXP_DIR

TMP_DIR=`mktemp -d`
mkdir $TMP_DIR/code
# symlink the data from the main dir

cat > $TMP_DIR/exp_train.sh <<EOF
#!/bin/bash -l
## Job name
#SBATCH -J ${EXP_TAG}_${NAME}
## Nodes
#SBATCH -N 1
## CPU per Node
#SBATCH -c 16
## GPU
#SBATCH --gres=gpu:a100-10:1
##
#SBATCH --mem=0
##
#SBATCH --time=670:00:00
##
#SBATCH --partition=mundus
##
#SBATCH --output="$EXP_DIR/exp_%j.out"
##
#SBATCH --error="$EXP_DIR/exp_%j.out"

## go to the exp dir
cd "$EXP_DIR/code"

/bin/hostname

eval "\$($REMOTE_MINICONDA_DIR/bin/conda shell.bash hook)"
conda activate cpc37

set -e
set -x

export PYTHONPATH=$EXP_DIR/code

python -u cpc/train.py \
    --pathCheckpoint $EXP_DIR \
    --pathDB ${REMOTE_DATA_DIR} --file_extension ${REMOTE_DATA_EXT} \
    --pathTrain ${REMOTE_DATA_SPLITS}/train_split.txt \
    --pathVal ${REMOTE_DATA_SPLITS}/test_split.txt \
    --n_process_loader 8 --max_size_loaded 4000000000 --batchSizeGPU 32 \
    --normMode layerNorm --dropout --rnnMode transformer  --nLevelsGRU 4  \
    --schedulerRamp 10 --nEpoch 30 \
    --CPCCTC --limitNegsInBatch 8  $PARAMS

CP=\$(ls $EXP_DIR/checkpoint*.pt | sed -e 's/.*_\([0-9]\+\).pt/\1/' | sort -n | tail -1)
mkdir -p $EXP_DIR/lineval_\${CP}
python -u cpc/eval/linear_separability.py \
    ${REMOTE_LIBRISPEECH_DIR}/train-clean-100 \
    ${REMOTE_LIBRISPEECH100_SPLITS}/train_split.txt \
    ${REMOTE_LIBRISPEECH100_SPLITS}/test_split.txt \
    $EXP_DIR/checkpoint_\${CP}.pt \
    --pathPhone ${REMOTE_LIBRISPEECH100_SPLITS}/converted_aligned_phones.txt \
    --file_extension .wav \
    --pathCheckpoint $EXP_DIR/lineval_\${CP} \
    2>&1 | tee -ai $EXP_DIR/lineval_\${CP}/out.txt
EOF

# Transmit the startup script
rsync $RVERB -lrpt -e "ssh -q" $TMP_DIR/ $REMOTE_USER@$REMOTE_HOST:$EXP_DIR/

# Transmit the rest
rsync --exclude '.*' \
      --exclude data \
      --exclude pretrained_models \
      --exclude '__pycache__' \
      --exclude '*runs*' \
      --exclude '*.pyc' \
      --exclude '*.ipynb' \
      --filter=':- .gitignore' \
    $RVERB -lrpt -e "ssh -q" $CPC_DIR/ $REMOTE_USER@$REMOTE_HOST:$EXP_DIR/code/

ssh -q $REMOTE_USER@$REMOTE_HOST sbatch `--mem=0 --time=670:00:00 --partition=mundus` \
    $EXP_DIR/exp_train.sh

rm -Rf $TMP_DIR

done

echo "Queue status"
ssh -q $REMOTE_USER@$REMOTE_HOST squeue
