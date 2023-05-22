# alpaco_5th_3
알파코 5기 3조 장기프로젝트 공유 레포입니다. 


### 개요
- Project 설명
- 모델설명
- Dataset
- Train 결과
- Fine-tuning
- Reference 

### PROJECT 설명 

*추후 작성 예정 * 
---

### ABOUT MODEL

---

### Dataset

+  AI hub / 한국어 음성 (입 모양 ) 영상 + 음성 / [You can Download Here!](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=538)

---
## TRAIN 

|CPU | GPU | RAM 
|------------|------|-------|
|i13- 12100f | RTX-3070Ti | 32G| 
|AWS | --| --|
---
__Wav2LIP__

<img src="https://user-images.githubusercontent.com/101646531/235811260-f4def410-14ec-406f-a0c4-c68fb31c0fed.gif" width="300" height="200"/> <img src="https://user-images.githubusercontent.com/101646531/235811264-d298537e-8a68-42a9-b8f0-f5395f2bfb7a.gif" width="300" height="200"/>


MakeItTalk


![배성재_makeittalk_kor](https://user-images.githubusercontent.com/121469546/235813171-b01d5e9c-4f2f-4c81-93d9-0818c5b4bf73.gif)
![배성재_makeittalk_eng](https://user-images.githubusercontent.com/121469546/235813155-73a2b65a-10da-4e75-afa7-9f9859a0f5a3.gif)



## data
```python
├── data/ # Kaldi-style data directory
│   ├── dev/        # validation set
│   ├── eval1/      # evaluation set
│   └── tr_no_dev/  # training set
├── dump/ # feature dump directory
│   ├── token_list/    # token list (dictionary)
│   └── raw/
│       ├── org/
│       │    ├── tr_no_dev/ # training set before filtering
│       │    └── dev/       # validation set before filtering
│       ├── srctexts   # text to create token list
│       ├── eval1/     # evaluation set
│       ├── dev/       # validation set after filtering
│       └── tr_no_dev/ # training set after filtering
└── exp/ # experiment directory
    ├── tts_stats_raw_phn_tacotron_g2p_en_no_space # statistics
    └── tts_train_raw_phn_tacotron_g2p_en_no_space # model
        ├── att_ws/                # attention plot during training
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── decode_train.loss.ave/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval1/ # evaluation set
        │        ├── att_ws/      # attention plot in decoding
        │        ├── probs/       # stop probability plot in decoding
        │        ├── norm/        # generated features
        │        ├── denorm/      # generated denormalized features
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── log/         # log directory
        │        ├── durations    # duration of each input tokens
        │        ├── feats_type   # feature type
        │        ├── focus_rates  # focus rate
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_5best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```
---

## ReFerence 


|Git|paper|
|---|-----|
|[wav_2lip](https://github.com/Rudrabha/Wav2Lip)| [paper](https://arxiv.org/pdf/2008.10010v1.pdf)|
[MakeItTalk](https://github.com/yzhou359/MakeItTalk) | [paper](https://arxiv.org/pdf/2004.12992v3.pdf)
