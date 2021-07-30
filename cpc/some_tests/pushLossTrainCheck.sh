

# this is just debug end-to-end check (debug option takes very small dataset subset in addition to printouts)

RUN_NAME="centerp_check001"

python ../train.py --pathDB /pio/gluster/data/ls-train-clean-100/LibriSpeech/train-clean-100 \
--pathTrain /pio/gluster/data/ls-train-clean-100/train_split.txt \
--pathVal /pio/gluster/data/ls-train-clean-100/test_split.txt \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
--max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
--schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
--supervised_classif_metric \
--path_phone_data /pio/gluster/data/ls-train-clean-100/converted_aligned_phones.txt \
--linsepBatchSizeGPU 8 --linsep_n_epoch 2 --linsep_times 1 \
--linsep_logs_dir /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}/linsep/logs \
--linsep_checkpoint_dir /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}/linsep/checkp \
--linsep_classif_each_epochs 2 \
--pathCheckpoint /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}/checkp/ \
--nEpoch 3 \
--modSettings --modCentermodule \
--modProtos 5 --modPushLossWeightEnc 0.3 --modPushLossLinear \
--modCenter_mode onlineKmeans --modCenter_onlineKmeansBatches 5 \
--modCenter_kmeansInitIters 1 --modCenter_kmeansInitBatches 1 \
--modCenter_initAfterEpoch 1 \
--modPushLossCenterNorm --modPushLossPointNorm --modCenter_norm --modPushLossNormReweight \
--overrideArgsFile \
--debug \
2>&1 | tee -ai /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}.log











