#!/bin/bash

# models
# bash train_ls100+buckeye.sh --pathCheckpoint /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-ls100-buckeye-linOut-encSegments \
# --ignore_cache --rnnModeRegHead none --rnnModeDownHead none --NoARonRegHead --normMode layerNorm --n_process_loader 1 \
# --limitNegsInBatch 8 --linearOutput --simMeasure cosine --smartPooling --samplingType samesequence --negativeSamplingExt 1 --nPredicts 1 \
# --CPCCTC --CPCCTCNumMatched 1 --CPCCTCNumLevels 2 --batchSizeGPU 32 --nEpoch 50 --nGPU 2 --nLevelsGRU 2 --schedulerRamp 10 \
# --segmentationType kreuk --encodeSegments 


### SANTIAGO

# Frame-wise phone classification accuracy
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/hacpc-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt --get_encoded --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-baseline-cosine-norelu-ls+buckeye/checkpoint_49.pt --get_encoded --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/cpc-kreuk-1neg-ls+buckeye-cosine-norelu/checkpoint_49.pt --get_encoded --ignore_cache

# CTC PER
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/hacpc-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-baseline-cosine-norelu-ls+buckeye/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/cpc-kreuk-1neg-ls+buckeye-cosine-norelu/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/hacpc-gt-cosine-norelu-encodeseg-ls/lineval_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-baseline-cosine-norelu-ls+buckeye/lineval_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/cpc-kreuk-1neg-ls+buckeye-cosine-norelu/lineval_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER --ignore_cache

# Phone segmentation
bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/hacpc-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt --boundaryDetector kreuk --get_encoded
bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-baseline-cosine-norelu-ls+buckeye/checkpoint_49.pt --boundaryDetector kreuk --get_encoded
bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/cpc-kreuk-1neg-ls+buckeye-cosine-norelu/checkpoint_49.pt --boundaryDetector kreuk --get_encoded

# Word segmentation
bash wordseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/hacpc-gt-cosine-norelu-encodeseg-ls/checkpoint_49.pt --boundaryDetector kreuk


### MACIEJ

### TRAINED ON LS/LS + BUCKEYE ###

# Frame-wise phone classification accuracy
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-ls100-linOut-encSegments/checkpoint_49.pt --get_encoded --ignore_cache
# bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-ls100-buckeye-linOut-encSegments/checkpoint_49.pt --get_encoded --ignore_cache

# CTC PER
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-ls100-linOut-encSegments/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank --ignore_cache
# bash lineval_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-ls100-buckeye-linOut-encSegments/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank --ignore_cache
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-ls100-linOut-encSegments/lineval_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER --ignore_cache
# bash lineval_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-ls100-buckeye-linOut-encSegments/lineval_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER --ignore_cache

# Phone segmentation
bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-ls100-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk --get_encoded
# bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-ls100-buckeye-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk --get_encoded

# Word segmentation
bash wordseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-ls100-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk
# bash wordseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-ls100-buckeye-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk


### TRAINED ON BUCKEYE ###

# Frame-wise phone classification accuracy
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --get_encoded --ignore_cache
# bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --get_encoded --ignore_cache


# CTC PER
# bash lineval_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --get_encoded --CTC --CTC_forbid_blank --ignore_cache
# bash lineval_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-buckeye-linOut-encSegments/linevalBuckeye_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER
bash lineval_ls100.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-buckeye-linOut-encSegments/lineval_ctc_noblank_onenc_2layers_49/checkpoint_9.pt --PER --ignore_cache

# Phone segmentation
# bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk --get_encoded
bash phoneseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk --get_encoded

# Word segmentation
# bash wordseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-kreukseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk
bash wordseg_buckeye.sh /pio/scratch/1/i325922/wav2vec/runs/cpc/acpc-hierarchical-GTseg-zelazkoarch-cosine-buckeye-linOut-encSegments/checkpoint_49.pt --boundaryDetector kreuk
