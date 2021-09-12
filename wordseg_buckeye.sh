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
  parser.add_argument('--boundaryDetector', type=str, help="Path to the checkpoint to evaluate.")
  parser.add_argument('--get_encoded', action='store_true')
  args, _ = parser.parse_known_args()

  checkpoint_dir = os.path.dirname(args.load)
  checkpoint_no = args.load.split('_')[-1][:-3]
  pathCheckpoint = f"{checkpoint_dir}/wordSegBuckeye{args.boundaryDetector}_{checkpoint_no}"
  if args.get_encoded:
    pathCheckpoint += '_onEnc'
  print(pathCheckpoint)
END
)"
mkdir -p ${SAVE_DIR}
echo $0 "$@" >> ${SAVE_DIR}/out.txt
exec python -u cpc/eval/segmentation.py \
    --pathDB /pio/scratch/1/i323106/data/BUCKEYE/test/ \
    --load "$@" \
    --pathPhone /pio/scratch/1/i323106/data/BUCKEYE/converted_aligned_phones.txt \
    --pathWords /pio/scratch/1/i325922/data/BUCKEYE/raw/converted_aligned_words.txt \
    --file_extension .wav \
    --pathCheckpoint $SAVE_DIR \
    2>&1 | tee -ai ${SAVE_DIR}/out.txt
