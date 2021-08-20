# Code and train logs for master's thesis "Enhancing unsupervised representation predictive coding with grouping-based approaches"


author: Piotr Pusz

thesis advisor: dr hab. inż. Jan Chorowski

This repo branch contains source conde and train logs for my masters thesis "Enhancing unsupervised representation predictive coding with grouping-based approaches". Source code is based on https://github.com/chorowski-lab/CPC_audio which builds on top of https://github.com/facebookresearch/CPC_audio/. It is written in Python 3.7 and uses popular libraries incl. PyTorch 1.8, numpy, matplotlib.

The code requires a GPU to run, all of the trainings in the thesis were performed in 2-GPU configuration.

## Setup instructions

1. install miniconda / conda
2. create a conda virtual env:
   ```
   conda create -n pusz_mgr_env python=3.7
   conda activate pusz_mgr_env
   ```
   (later instructions assuming virtual environment activated)
3. install pytorch 1.8 (current LTS release):
   `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia`
4. install needed packages:
   ```
   conda install psutil tqdm nose cython
   python -m pip install soundfile progressbar2 matplotlib
   ```
5. clone this repo and choose the branch:
   ```
   git clone https://github.com/chorowski-lab/CPC_audio.git
   git fetch
   git checkout pp/thesis
   ```
6. setup this repo as a package from the cpc directory:
   ```
   cd CPC_audio
   python setup.py develop
   ```
7. download LibriSpeech train-clean-100 data e.g. in .flac file format and place under `LS-TC100` directory
8. download LibriSpeech train-clean-100 test/val split files and phoneme alignment data from https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb (as in the original CPC_audio repo). I will denote paths to those files (not directories, but whole paths with file names) as `LS-TC100-TRAIN`, `LS-TC100-VAL` and `LS-ALIGNMENTS`.
9. Create a directory `TRAINING-ROOT` for training data


## How to train the model with modifications presented in the thesis

Main runnable train script is the `cpc/train.py` file.

In order to use fixed random seed for the runs (as I did for many experiments in the thesis), use `--random_seed` parameter.

Description of parameters not mentioned here can be found in `cpc/train.py` or its help dialog if needed; the default values for the parameters can be found in `cpc/train.py` and `cpc/cpc_default_config.py` files.

This readme uses notation shortcuts present in the thesis.

### Baseline

Example run configuration for 50 epochs, with automatic phoneme linear separability training performed 3 times after unsupervised model training:
```
RUN_NAME=some_baseline_50

python train.py --pathDB $LS-TC100 \
--pathTrain $LS-TC100-TRAIN \
--pathVal $LS-TC100-VAL \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
--max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
--schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
--supervised_classif_metric \
--path_phone_data $LS-ALIGNMENTS \
--linsepBatchSizeGPU 8 --linsep_n_epoch 10 --linsep_times 3 \
--linsep_logs_dir ${TRAINING-ROOT}/${RUN_NAME}/linsep/logs \
--linsep_checkpoint_dir ${TRAINING-ROOT}/${RUN_NAME}/linsep/checkp \
--linsep_classif_each_epochs 49 \
--pathCheckpoint ${TRAINING-ROOT}/${RUN_NAME}/checkp/ \
--nEpoch 50 \
2>&1 | tee -ai ${TRAINING-ROOT}/${RUN_NAME}.log
```

Description of the parameters used can be found in `train.py` file or in its help dialog.

### Centroid-based denoising

Run parameters corresponding to centroid-based denoising which are used in the thesis (with (*) denoting parameters essential to run centroid-based denosing at all) are:
- `--modSettings` (*) - enables the use of the modifications from the thesis
- `--modProtos` (*) - number of centroids to use
- `--modPushLossWeightEnc` (*) - weight of centroid-based denoising loss term on representations z ('encodings')
- `--modPushLossLinear` - use linear centroid-based denoising distance loss term instead of square one
- `--modCenter_norm`, `--modPushLossCenterNorm`, `--modPushLossPointNorm` (use together) - normalize representations for centroid-based denoising
- `--modPushLossNormReweight` - loss reweighting for normalization, r-avg
- `--modPushLossReweightPointsSeparately` - loss reweighting for normalization, r-sep
- `--modCentermodule` (*) - use module performing online k-means for centroid-based denoising
- `--modCenter_mode` (*) - mode how to run centroid estimation; onlineKmeans for online k-means used in the thesis
- `--modCenter_initAfterEpoch` (*) - after which epoch initialize the centroids; needs to be at least 2 epoch after the initial state (begin or checkpoint resumed)
- `--modCenter_kmeansInitIters` (*) - number regular k-means iterations for initializing the centroids for online k-means
- `--modCenter_kmeansInitBatches` (*) - number of batches on which regular k-means which initializes online k-means is run; this is a number of points drawn (for each point its whole batch is taken), a few batches can repreat
- `--modCenter_kmeansReinitEachN` - how frequently in terms of epochs perform online k-means reinitialization
- `--modCenter_kmeansReinitUpTo` - upper bound on the epoch number when online k-means can be reinitialized, to avoid reinitializing near training end
- `--modCenter_onlineKmeansBatches` (*) - length of online k-means memory window in batches

There are some additional parameters not used in the thesis which are described in `train.py` file or in its help dialog.

Configuration initializing centroid-based denoising at the beginning of the training (after epoch 2; because of implementation details it has to be at least after epoch 1) that achieved best scores with random seed:

```
RUN_NAME=some_good_centroid-based-denoising_config_50

python train.py --pathDB $LS-TC100 \
--pathTrain $LS-TC100-TRAIN \
--pathVal $LS-TC100-VAL \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
--max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
--schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
--supervised_classif_metric \
--path_phone_data $LS-ALIGNMENTS \
--linsepBatchSizeGPU 8 --linsep_n_epoch 10 --linsep_times 3 \
--linsep_logs_dir ${TRAINING-ROOT}/${RUN_NAME}/linsep/logs \
--linsep_checkpoint_dir ${TRAINING-ROOT}/${RUN_NAME}/linsep/checkp \
--linsep_classif_each_epochs 49 \
--pathCheckpoint ${TRAINING-ROOT}/${RUN_NAME}/checkp/ \
--nEpoch 50 \
--modSettings --modCentermodule \
--modProtos 100 --modPushLossWeightEnc 0.3 --modPushLossLinear \
--modCenter_mode onlineKmeans --modCenter_onlineKmeansBatches 3500 \
--modCenter_kmeansInitIters 50 --modCenter_kmeansInitBatches 200 \
--modCenter_initAfterEpoch 2 \
--modPushLossCenterNorm --modPushLossPointNorm --modCenter_norm --modPushLossReweightPointsSeparately \
2>&1 | tee -ai ${TRAINING-ROOT}/${RUN_NAME}.log
```

In order to add centroid-based denoising after some epoch, using a trained checkpoint, create the `${TRAINING-ROOT}/${RUN_NAME}/checkp/` directory and copy the checkpoint there. In the run configuration, specify `--modCenter_initAfterEpoch` not less than checkpoint epoch plus 2. Then to use same seed, either manually specify the same random seed used in the previous training or copy `checkpoint_args.json` file from the previous run and add the `--overrideArgsFile` option to a new run file (common basic arguments like random seed will be copied from previous file and `--overrideArgsFile` will override other parameters with ones specified in the new run file if there is a conflict, leaving random seed used unchanged).


### PDACPC

Run parameters corresponding to PDACPC which are used in the thesis (with (*) denoting parameters essential to run centroid-based denosing at all) are:
- `modSettings` (*) - enables the use of the modifications from the thesis
- `nPredictorsTimeAligned` - number of predictors used in PDACPC
- `modelLengthInARsimple` (*, or `--modelLengthInARconv`) - use regular frame length modeling in terms of phoneme duration for PDACPC model
- `modelLengthInARconv` (*, or `--modelLengthInARsimple`) - use frame length modeling in terms of phoneme duration with conv layer (allowing to see future representations, only for length prediction) for PDACPC model
- `ARmap01rangeMin` (*) - lower range of interval to which predicted frame len is mapped in PDACPC
- `ARmap01rangeMax` (*) - upper range of interval to which predicted frame len is mapped in PDACPC
- `modelLengthInARweightsMode` (*) - type of weights used in PDACPC model - exp, doubleExp, bilin, trilin or normals
- `modelLengthInARweightsCoeff` (* if exp or doubleExp mode) - alpha coefficient for exp, doubleExp weights in PDACPC
- `ARmodelFrameNormalsSigma` (* if normals mode) - sigma parameter for PDACPC weights: normals, exp, doubleExp lin sigma (here it is 'trilin' mode with sigma)
- `ARteachOnlyLastFrameLength` - rebalancing loss teaching frame lengths - onlyTeachLast / OTL
- `ARteachLongPredsUniformlyLess` - rebalancing loss teaching frame lengths - teachUniformlyLess / TUL
- `ARteachLongPredsSqrtLess` - rebalancing loss teaching frame lengths - teachSqrtLess / TSL
- `ARlengthsGradReweight` - gradient reqeighting coefficient for rebalancing loss teaching frame lengths
- `ARlengthFirstPredID` - 'ID' variation of PDACPC - change first predictor for identity with current frame; can use with increasing --nPredictorsTimeAligned by 1
- `ARlengthPredNoise` - standard deviation of normal noise added to predicted lengths in PDACPC - BEFORE MAPPING (e.g. with --modelLengthInARsimple and len 0.4-0.6, 0.1 will result in 0.01 after mapping)
- `predShowDetachedLengths` - show predicted frame lengths (with stopGradient applied) in predictors input
- `linsepShowARlengthsInCtx` - show predicted frame lengths with stopGradient applied in context representations fed to linear separability model

Configuration with PDACPC that achieved best scores with random seed:

```
RUN_NAME=some_good_centroid-based-denoising_config_50

python train.py --pathDB $LS-TC100 \
--pathTrain $LS-TC100-TRAIN \
--pathVal $LS-TC100-VAL \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
--max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
--schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
--supervised_classif_metric \
--path_phone_data $LS-ALIGNMENTS \
--linsepBatchSizeGPU 8 --linsep_n_epoch 10 --linsep_times 3 \
--linsep_logs_dir ${TRAINING-ROOT}/${RUN_NAME}/linsep/logs \
--linsep_checkpoint_dir ${TRAINING-ROOT}/${RUN_NAME}/linsep/checkp \
--linsep_classif_each_epochs 49 \
--pathCheckpoint ${TRAINING-ROOT}/${RUN_NAME}/checkp/ \
--nEpoch 50 \
--modSettings --modelLengthInARsimple --modelLengthInARweightsMode normals --ARmodelFrameNormalsSigma 0.1 \
--ARmap01rangeMin 0.4 --ARmap01rangeMax 0.6 \
--ARteachLongPredsUniformlyLess \
--predShowDetachedLengths --linsepShowARlengthsInCtx --nPredictorsTimeAligned 12 \
2>&1 | tee -ai ${TRAINING-ROOT}/${RUN_NAME}.log
```

### Hierarchical segmentation of representations (not presented in main thesis text)

This modification isn't presented in main thesis text as it didn't achieve satisfying results; it involves hierarchical segmentation of representations z before feeding them to the model producing context representations c. Its not satisfying results may be because of model shifting data across time which decreases framewise linear separability performance but could work well in other measures, so I mention it here shortly for reference.

Main parameters used by this hierarchical segmentation are:
- `--modHierARshorten`
- `--modHierARgradualStart`
- `--modHierARmergePrior`
- `--modSegmentCostModule`
- `--modSegment_batchesMem`

## Repo structure

All of the thesis-specific code not inherited from the parent repositories is under the `cpc` folder.

Code for centroid-based denoising is present in `cpc/train.py`, `cpc/model.py`, `center_model.py` (online k-means).

Code for PDACPC is present in `cpc/criterion/soft_align.py`, `cpc/train.py`, `cpc/model.py`.

Jupyter notebook used to create visualization figures in the thesis is under `cpc/criterion/visualization.ipynb`.

Code for hierarchical segmentation is present in `cpc/train.py`, `cpc/model.py`, and in the `cpc/segm` folder.

## Tests

Runnable tests for centroid-based denoising and PDACPC modifications are present in the `some_tests` directory.
Tests for hierarchical segmentation of representations are in their source code files when run as python modules.

## Train logs

Output logs from the trainings results of which are presented in the thesis text can be found under `logs` as an archive file (in the snapshot of this repository uploaded to APD those are removed because of the size - please refer to the full repo branch at https://github.com/chorowski-lab/CPC_audio/tree/pp/thesis). In each log file name prefix, position of the configuration corresponding to this run in the tables presented in the thesis is indicated (with 0- or 1-based numeration of rows and columns); however few of the run file names may contain misleading parts as I made some mistakes in naming - actual parameters are always output in the log file content near its beginning and are same as presented in the thesis text.

--

Piotr Pusz


original readme from parent repository below

----------------------------- 



# CPC_audio

This code implements the Contrast Predictive Coding algorithm on audio data, as described in the paper [Unsupervised Pretraining Transfers well Across Languages](https://arxiv.org/abs/2002.02848). This is an unsupervised method to train audio features directly from the raw waveform.

Moreover, this code also implements all the evaluation metrics used in the paper:
- [ABX discriminability](https://zerospeech.com/2017/track_1.html)
- [Phone and speaker linear separability](https://arxiv.org/abs/1807.03748)
- Transfer learning on other languages, using the [common voices datasets](https://voice.mozilla.org/en/datasets)

## Setup instructions

The installation is a tiny bit involved due to the torch-audio dependency.

0/ Clone the repo:
`git clone git@github.com:facebookresearch/CPC_audio.git && cd CPC_audio`

1/ Install libraries which would be required for torch-audio https://github.com/pytorch/audio :
 * MacOS: `brew install sox`
 * Linux: `sudo apt-get install sox libsox-dev libsox-fmt-all`

2/ `conda env create -f environment.yml && conda activate cpc37`

3/ Run setup.py
`python setup.py develop`

You can test your installation with:
`nosetests -d`

### CUDA driver

This setup is given for CUDA 9.2 if you use a different version of CUDA then please change the version of cudatoolkit in environment.yml.
For more information on the cudatoolkit version to use, please check https://pytorch.org/

### Standard datasets

We suggest to train the model either on [Librispeech](http://www.openslr.org/12/) or [libri-light](https://github.com/facebookresearch/libri-light).


## How to run a session

To run a new training session, use:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
```

Where:
- $PATH_AUDIO_FILES is the directory containing the audio files. The files should be arranged as below:
```
PATH_AUDIO_FILES  
│
└───speaker1
│   └───...
│         │   seq_11.{$EXTENSION}
│         │   seq_12.{$EXTENSION}
│         │   ...
│   
└───speaker2
    └───...
          │   seq_21.{$EXTENSION}
          │   seq_22.{$EXTENSION}
```

Please note that each speaker directory can contain an arbitrary number of subdirectories: the speaker label will always be retrieved from the top one. The name of the files isn't relevant. For a concrete example, you can look at the organization of the [Librispeech](http://www.openslr.org/12/) dataset.

- $PATH_CHECKPOINT_DIR in the directory where the checkpoints will be saved
- $TRAINING_SET is a path to a .txt file containing the list of the training sequences (see [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb) for example)
- $VALIDATION_SET is a path to a .txt file containing the list of the validation sequences
- $EXTENSION is the extension of each audio file

## Custom architectures

The code allows you to train a wide range of architectures. For example, to train the CPC method as described in [Van Den Oord's paper](https://arxiv.org/abs/1807.03748) just run:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --normMode batchNorm --rnnMode linear
```

Or if you want to train a model with a FFD prediction network instead of a transformer:
```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --rnnMode ffd --schedulerRamp 10
```

The --schedulerRamp option add a learning rate ramp at the beginning of the training: it barely affects the performance of a model with a transformer predictor but is necessary with other models.

Launch cpc/train.py -h to see all the possible options.

## How to restart a session

To restart a session from the last saved checkpoint just run
```bash
python cpc/train.py --pathCheckpoint $PATH_CHECKPOINT_DIR
```
## How to run an evaluation session

All evaluation scripts can be found in cpc/eval/.

### Linear separability:

After training, the CPC model can output high level features for a variety of tasks. For an input audio file sampled at 16kHz, the provided baseline model will output 256 dimensional output features every 10ms. We provide two linear separability tests one for speaker, one for phonemes, in which a linear classifier is trained on top of the CPC features with aligned labels, and evaluated on a held-out test set.

Train / Val splits as well as phone alignments for librispeech-100h can be found [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb).


Speaker separability:

```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT
```

Phone separability:
```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --pathPhone $PATH_TO_PHONE_LABELS
```

You can also concatenate the output features of several model by providing several checkpoint to the --load option. For example the following command line:

```bash
python cpc/eval/linear_separability.py -$PATH_DB $TRAINING_SET $VAL_SET model1.pt model2.pt --pathCheckpoint $PATH_CHECKPOINT
```

Will evaluate the speaker separability of the concatenation of the features from model1 and model2.

`--gru_level` controls from which layer of autoregressive part of CPC to extract the features. By default it's the last one.

Nullspaces:

To conduct the nullspace experiment, first classify speakers using two factorized matrices `A` (`DIM_EMBEDDING` x `DIM_INBETWEEN`) and `B` (`DIM_INBETWEEN` x `SPEAKERS`). You'll want to extract `A'`, the nullspace of matrix `A` (of size `DIM_EMBEDDING` x (`DIM_EMBEDDING` - `DIM_INBETWEEN`)), to make the embeddings less sensitive to speakers. 
```bash 
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --mode speakers_factorized  --model cpc --dim_inter $DIM_INBETWEEN --gru_level 2
```

Next, you evaluate the phone and speaker separabilities of the embeddings from CPC projected into the nullspace `A'`.
```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --mode phonemes_nullspace --model cpc --pathPhone $PATH_TO_PHONE_LABELS --path_speakers_factorized $PATH_CHECKPOINT_SPEAKERS_FACTORIZED --dim_inter $DIM_INBETWEEN --gru_level 2
```

```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --mode speakers_nullspace --model cpc --path_speakers_factorized $PATH_CHECKPOINT_SPEAKERS_FACTORIZED --dim_inter $DIM_INBETWEEN --gru_level 2
```

### ABX score:

You can run the ABX score on the [Zerospeech2017 dataset](https://zerospeech.com/2017/index.html). To begin, download the dataset [here](https://download.zerospeech.com/). Then run the ABX evaluation on a given checkpoint with:

```bash
python ABX.py from_checkpoint $PATH_CHECKPOINT $PATH_ITEM_FILE $DATASET_PATH --seq_norm --strict --file_extension .wav --out $PATH_OUT
```
Where:
- $PATH_CHECKPOINT is the path pointing to the checkpoint to evaluate
- $PATH_ITEM_FILE is the path to the .item file containing the triplet annotations
- $DATASET_PATH path to the directory containing the audio files
- $PATH_OUT path to the directory into which the results should be dumped
- --seq_norm normalize each batch of features across the time channel before computing ABX
- --strict forces each batch of features to contain exactly the same number of frames.

### Cross lingual transfer

To begin download the common voices datasets [here](https://voice.mozilla.org/en/datasets), you will also need to download our phonem annotations and our train / val / test splits for each language [here](https://dl.fbaipublicfiles.com/cpc_audio/common_voices_splits.tar.gz). Then unzip your data at PATH_COMMON_VOICES.
Unfortunately, the audio files in common voices don't have the same sampling rate as in Librispeech. Thus you'll need to convert them into 16kH audio using the command:

```bash
DIR_CC=$PATH_COMMON_VOICES
for x in fr zh it ru nl sv es tr tt ky; do python cpc/eval/utils/adjust_sample_rate.py ${DIR_CC}/${x}/clips ${DIR_CC}/${x}/validated_phones_reduced.txt ${DIR_CC}/${x}/clips_16k; done
```

You can now run the experiments described in the paper. To begin, you must train the linear classifier. You will find below the instructions for the Spanish dataset: you can run the experiments on any other dataset in the same fashion.

#### Frozen features

To run the training on frozen features with the one hour dataset, just run:

```bash
python cpc/eval/common_voices_eval.py train $PATH_COMMON_VOICES/es/clips_16k $PATH_COMMON_VOICES/es/validated_phones_reduced.txt $CHECKPOINT_TO_TEST --pathTrain $PATH_COMMON_VOICES/es/trainSeqs_1.0_uniform_new_version.txt  --pathVal $PATH_COMMON_VOICES/es/trainSeqs_1.0_uniform_new_version.txt --freeze -o $OUTPUT_DIR
```

#### Fine tuning

The command is quite similar to run the fine-tuning experiments on the 5 hours dataset. For example in French you need to run:
```bash
python cpc/eval/common_voices_eval.py train $PATH_COMMON_VOICES/es/clips_16k $PATH_COMMON_VOICES/es/validated_phones_reduced.txt $CHECKPOINT_TO_TEST --pathTrain $PATH_COMMON_VOICES/es/trainSeqs_5.0_uniform_new_version.txt --pathVal $PATH_COMMON_VOICES/es/trainSeqs_5.0_uniform_new_version.txt --freeze -o $OUTPUT_DIR
```

#### PER

Once the training is done, you can compute the associated phone error rate (PER) on the test subset. To do so, just run:

```bash
python cpc/eval/common_voices_eval.py per $OUTPUT_DIR --pathVal $PATH_COMMON_VOICES/es/testSeqs_uniform_new_version.txt --pathPhone $PATH_COMMON_VOICES/es/validated_phones_reduced.txt
```

## torch hub

To begin download the common voices datasets [here](https://voice.mozilla.org/en/datasets), you will also need to download our phonem annotations and our train / val / test splits for each language [here](https://dl.fbaipublicfiles.com/cpc_audio/common_voices_splits.tar.gz). Then unzip your data at PATH_COMMON_VOICES.
Unfortunately, the audio files in common voices don't have the same sampling rate as in Librispeech. Thus you'll need to convert them into 16kH audio using the command:

```bash
DIR_CC=$PATH_COMMON_VOICES
for x in fr zh it ru nl sv es tr tt ky; do python cpc/eval/utils/adjust_sample_rate.py ${DIR_CC}/${x}/clips ${DIR_CC}/${x}/validated_phones_reduced.txt ${DIR_CC}/${x}/clips_16k; done
```

You can now run the experiments described in the paper. To begin, you must train the linear classifier. You will find below the instructions for the Spanish dataset: you can run the experiments on any other dataset in the same fashion.

#### Frozen features

To run the training on frozen features with the one hour dataset, just run:

```bash
python cpc/eval/common_voices_eval.py train $PATH_COMMON_VOICES/es/clips_16k $PATH_COMMON_VOICES/es/validated_phones_reduced.txt $CHECKPOINT_TO_TEST --pathTrain $PATH_COMMON_VOICES/es/trainSeqs_1.0_uniform_new_version.txt  --pathVal $PATH_COMMON_VOICES/es/trainSeqs_1.0_uniform_new_version.txt --freeze -o $OUTPUT_DIR
```

#### Fine tuning

The command is quite similar to run the fine-tuning experiments on the 5 hours dataset. For example in French you need to run:
```bash
python cpc/eval/common_voices_eval.py train $PATH_COMMON_VOICES/es/clips_16k $PATH_COMMON_VOICES/es/validated_phones_reduced.txt $CHECKPOINT_TO_TEST --pathTrain $PATH_COMMON_VOICES/es/trainSeqs_5.0_uniform_new_version.txt --pathVal $PATH_COMMON_VOICES/es/trainSeqs_5.0_uniform_new_version.txt --freeze -o $OUTPUT_DIR
```

#### PER

Once the training is done, you can compute the associated phone error rate (PER) on the test subset. To do so, just run:

```bash
python cpc/eval/common_voices_eval.py per $OUTPUT_DIR --pathVal $PATH_COMMON_VOICES/es/testSeqs_uniform_new_version.txt --pathPhone $PATH_COMMON_VOICES/es/validated_phones_reduced.txt
```

## torch hub

This model is also available via [torch.hub](https://pytorch.org/docs/stable/hub.html). For more details, have a look at hubconf.py.

## Citations
Please consider citing this project in your publications if it helps your research.

```
@misc{rivire2020unsupervised,
    title={Unsupervised pretraining transfers well across languages},
    author={Morgane Rivière and Armand Joulin and Pierre-Emmanuel Mazaré and Emmanuel Dupoux},
    year={2020},
    eprint={2002.02848},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

## License

CPC_audio is MIT licensed, as found in the LICENSE file.
