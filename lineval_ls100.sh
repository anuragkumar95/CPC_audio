#!/bin/bash

set -e
set -x

RVERB="-v --dry-run"
RVERB=""
CPC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAVE_DIR="$(
python - "$@" << END
if 1:
  import argparse
  import os.path
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('load', type=str,
                        help="Path to the checkpoint to evaluate.")
  parser.add_argument('--pathCheckpoint')
  parser.add_argument('--CTC', action='store_true')
  parser.add_argument('--get_encoded', action='store_true')
  args, _ = parser.parse_known_args()
  checkpoint_dir = os.path.dirname(args.load)
  checkpoint_no = args.load.split('_')[-1][:-3]
  eval_ctc = ""
  if args.CTC:
    eval_ctc = "_ctc"
  enc_str = ""
  if args.get_encoded:
    enc_str = "_onenc"
  print(f"{checkpoint_dir}/lineval{enc_str}{eval_ctc}_{checkpoint_no}")
END
)"

if [ -e ${SAVE_DIR}/checkpoint_9.pt ]; then
echo "skipping eval"
exit 0
fi

mkdir -p ${SAVE_DIR}/code
rsync --exclude '.*' \
      --exclude data \
      --exclude pretrained_models \
      --exclude '__pycache__' \
      --exclude '*runs*' \
      --exclude '*.pyc' \
      --exclude '*.ipynb' \
      --filter=':- .gitignore' \
    $RVERB -lrpt $CPC_DIR/ ${SAVE_DIR}/code/

echo $0 "$@" >> ${SAVE_DIR}/out.txt
export PYTHONPATH=$CPC_DIR:$PYTHONPATH
exec python -u cpc/eval/linear_separability.py \
    /pio/data/zerospeech2021/LibriSpeech-wav/train-clean-100 \
    /pio/scratch/1/jch/wav2vec/LibriSpeech100_labels_split/train_split.txt \
    /pio/scratch/1/jch/wav2vec/LibriSpeech100_labels_split/test_split.txt \
    "$@" \
    --pathPhone /pio/scratch/1/jch/wav2vec/LibriSpeech100_labels_split/converted_aligned_phones.txt \
    --file_extension .wav \
    --pathCheckpoint $SAVE_DIR \
    `#--max_size_loaded 1000000000` 2>&1 | tee -ai ${SAVE_DIR}/out.txt
 