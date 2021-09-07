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
  parser.add_argument('--CTC_forbid_blank', action='store_true')
  parser.add_argument('--numLayers', type=int, default=2)
  parser.add_argument('--CPCLevel', type=int, default=0, help="")
  args, _ = parser.parse_known_args()
  checkpoint_dir = os.path.dirname(args.load)
  checkpoint_no = args.load.split('_')[-1][:-3]
  eval_ctc = ""
  if args.CTC:
    eval_ctc += "_ctc"
  if args.CTC_forbid_blank:
    eval_ctc += "_noblank"
  if args.get_encoded:
    eval_ctc += "_onenc"
  if args.numLayers:
    eval_ctc += f"_{args.numLayers}layers"
  if args.CPCLevel > 0:
    eval_ctc += f"_level{args.CPCLevel}"
  print(f"{checkpoint_dir}/lineval{eval_ctc}_{checkpoint_no}")
END
)"

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
exec python -u cpc/eval/linear_separability.py \
    --pathDB /pio/gluster/data/ls-train-clean-100/LibriSpeech/train-clean-100 \
    --pathTrain /pio/gluster/data/ls-train-clean-100/train_split.txt \
    --pathVal /pio/gluster/data/ls-train-clean-100/test_split.txt \
    --load "$@" \
    --pathPhone /pio/gluster/data/ls-train-clean-100/converted_aligned_phones.txt \
    --file_extension .flac \
    --pathCheckpoint $SAVE_DIR \
    2>&1 | tee -ai ${SAVE_DIR}/out.txt