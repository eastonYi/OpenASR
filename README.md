# OpenASR

A pytorch based end2end speech recognition system. The main architecture is [Speech-Transformer](https://ieeexplore.ieee.org/abstract/document/8462506/).

[中文说明](https://github.com/by2101/OpenASR/blob/master/README_zh.md)

## Results

### aishell1

| Model | #Snt | #Wrd |   Sub  |  Del  | Ins  |  CER|
| :---: | :-: | :----: |:----: |:----: |:----: | :----: |
| Conv-Transformer | 7176 | 104765 | 6.02  |  0.24 |  0.17  | 6.44 |


### hkust
| Model | #Snt | #Wrd |   Sub  |  Del  | Ins  |  CER |
| :---: | :-: | :----: |:----: |:----: |:----: | :----: |
| Conv-Transformer | 5236 | 55894 | 19.41 | 3.05 |  3.90 |  26.35 |
| Conv-CTC  | 5236 | 55894 | 21.27 | 3.90 | 2.74 | 27.91 |

## Features

1. **Minimal Dependency**. The system does not depend on external softwares for feature extraction or decoding. Users just install PyTorch deep learning framework.
2. **Good Performance**. The system includes advanced algorithms, such as Label Smoothing, SpecAug, LST, and achieves good performance on ASHELL1. The baseline CER on AISHELL1 test is 6.6, which is better than ESPNet.
3. **Modular Design**. We divided the system into several modules, such as trainer, metric, schedule, models. It is easy for extension and adding features.
4. **End2End**. The feature extraction and tokenization are online. The system directly processes wave file. So, the procedure is much simpified.

## Dependency
* python >= 3.6
* pytorch >= 1.5
* pyyaml >= 5.1
* [ctcdecode](https://github.com/parlance/ctcdecode/) >= 0.7(optional)

## Usage
We use KALDI style example organization. The example directory include top-level shell scripts, data directory, exp directory. We provide an AISHELL-1 example. The path is ROOT/egs/aishell1/s5.

### Data Preparation
- aishell1
python ~/projects/OpenASR/tools/gen_json.py --feat feats.scp --num_frames utt2num_frames --trans text --output train.json

### Train Models
We use yaml files for parameter configuration. We provide 3 examples.

    config_base.yaml  # baseline ASR system
    config_lm_lstm.yaml  # LSTM language model
    config_lst.yaml  # training ASR with LST

Run train.sh script for training baseline system.

    bash train.sh

### Model Averaging
Average checkpoints for improving performance.

    bash avg.sh

### Decoding and Scoring
Run decode_test.sh script for decoding test set.

    bash decode_test.sh
    bash score.sh data/test/text exp/exp1/decode_test_avg10

## CPC
### DATA Prepare
python ~/projects/OpenASR/tools/gen_wav_flist.py --wav-dir test-clean  --ext flac --output test-clean.flist

## Acknowledgement
This system is implemented with PyTorch. We use wave reading codes from SciPy. We use SCTK software for scoring. Thanks to Dan Povey's team and their KALDI software. I learn ASR concept, and example organization from KALDI. And thanks to Google Lingvo Team. I learn the modular design from Lingvo.

## Bib
@article{bai2019learn,
  title={Learn Spelling from Teachers: Transferring Knowledge from Language Models to Sequence-to-Sequence Speech Recognition},
  author={Bai, Ye and Yi, Jiangyan and Tao, Jianhua and Tian, Zhengkun and Wen, Zhengqi},
  year={2019}
}

## References
Dong, Linhao, Shuang Xu, and Bo Xu. "Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
Zhou, Shiyu, et al. "Syllable-based sequence-to-sequence speech recognition with the transformer in mandarin chinese." arXiv preprint arXiv:1804.10752 (2018).
