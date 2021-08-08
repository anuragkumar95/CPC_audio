#
# Based on the `CPC-big-kmeans50/cpc_ll6k` ZeroSpeech2021 baseline model config
#

: ${DATASET_DIR:="/datasets/libri-light/original/output_dir"}  # has small/ medium/ large/ and dev-clean from LS
: ${OUTPUT_DIR:="results/acpc_big_ll60k_test"}
: ${BATCH_SIZE:=16}  # Per GPU for 32 GPUs

mkdir -p $OUTPUT_DIR

ARGS+=" --pathTrain filelists/ll-0.6k.txt"  # NOTE 0.6k only for debugging; use 6k or 60k
ARGS+=" --pathVal filelists/ls-dev-clean.txt"
ARGS+=" --batchSizeGPU $BATCH_SIZE"
# ARGS+=" --distributed"

# ACPC
ARGS+=" --nPredicts 8"  # NOTE 12 in the CPC big model
ARGS+=" --CPCCTC"
ARGS+=" --CPCCTCNumMatched 12"
ARGS+=" --CPCCTCSelfLoop"
ARGS+=" --CPCCTCSkipBeg 1"
ARGS+=" --CPCCTCSkipEnd 2"

# big
ARGS+=" --hiddenEncoder 512"
ARGS+=" --hiddenGar 512"
ARGS+=" --dropout"
ARGS+=" --multihead_rnn"
ARGS+=" --nLevelsGRU 4"
ARGS+=" --schedulerRamp 10"

# io
ARGS+=" --save_step 1"
ARGS+=" --loadCriterion"
ARGS+=" --path_cache ${OUTPUT_DIR}/_cpc_cache.txt"
ARGS+=" --pathDB $DATASET_DIR"
ARGS+=" --pathCheckpoint ${OUTPUT_DIR}"
ARGS+=" --file_extension .flac"
ARGS+=" --n_process_loader 8"
ARGS+=" --max_size_loaded 200000000"

python -u cpc/train.py $ARGS 2>&1 | tee -ai ${OUTPUT_DIR}/acpc_big.log
