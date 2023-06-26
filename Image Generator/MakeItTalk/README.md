# MIT TEMPLATE

Image Generation 모델인 MakeItTalk입니다.

## Table of Contents

* [MIT TEMPLATE](#mit-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [0\. Environment](#0-environment)
    * [1\. Data Information](#1-data-information)
    * [2\. Data preparation](#2-data-preparation)
    * [3\. Wav dump / Embedding preparation](#3-wav-dump--embedding-preparation)
    * [4\. Text dump / Scaling](#4-text-dump--scaling)
    * [5\. Training](#5-training)
    * [6\. Inference](#6-inference)
  * [Git directory map](#git-directory)

## Recipe flow

### 0. Environment
- Python environment 3.6

```sh
conda create -n makeittalk_env python=3.6
conda activate makeittalk_env
conda env update --file makeittalk_config.yml
```

### 1. Data Information

데이터 : 싱크가 일치하는 mp4, wav파일 데이터셋

### 2. Data preparation

landmark 추출 및 폴더 관리 : 비디오분할_facelandmark.ipynb

### 3. Wav dump / Embedding preparation

audio 파일 mel_spectrogram + Voice Conversion Embedding : mel_spectrogram.py

embedding 파일 scaling 및 피클화 : mel_spectrogram_ms.py

### 4. Text dump / Scaling

landmark text 파일 전처리 : txt_preprocessing.py

### 5. Training content module

위에서 만든 {}\_fl.pickle, {}\_au.pickle 파일을 src/approaches/train_audio2landmark.py dataloader 경로 설정 및 해당 파일명으로 변경

원하는 dump 파일에 옮기고 main_train_content.py 실행

```sh
# --dump_dir <dump pickle 경로>
# --name <ckpt>/<save_weights_path>
# --load_a2l_C_name <pretrained_model_path>
$ python main_train_content.py --train --write --dump_dir dump_6 \
--name sixth --ckpt_epoch_freq 3 --batch_size 16 \
--load_a2l_C_name examples/ckpt/ckpt_content_branch.pth
```

See also:
- [Training Source](https://github.com/adobe-research/MakeItTalk)

### 6. Inference

인퍼런스 코드 + 작성

## 7. result
결과 데모 및 training 표 작성

## Git directory 

```
├── doc/            # documents
├── facewarp/       # 카툰 캐릭터 이미지 생성
└── src/            # source code
    ├── approaches/ # training code
    ├── autovc/     # audio embedding code
    ├── dataset/    # dataset classes
    └── models/     # model structures
└── thirdparty/     # use code
    ├── AdaptiveWingLoss # loss function
    └── resemblyer_util  # for audio embedding
├── examples/                       # examples & ckpt
├── util/                           # utils folder
├── main_end2end_cartoon.py         # 카툰 이미지 end2end code
├── main_end2end.py                 # 실사 이미지 end2end code
├── main_train_content.py           # content module training code
├── main_train_image_translation.py # image translation training code
├── main_train_speaker_aware.py     # speaker_aware training code
├── makeittalk_config.yaml          # environment
├── mel_spectrogram_ms.py           # audio embedding Scaling 
├── mel_spectrogram.py              # audio embedding with autoVC
├── quick_demo.ipynb                # quick_demo notebook
├── requirements.txt                # quick environment
├── txt_preprocessing.py            # landmark text normalization
└── 비디오분할_facelandmark.ipynb    # 비디오 분할 notebook
```

## Reference

| Reference |  Link  | 
| :-------------: | :---------------: |
| MakeItTalk 원문 | [Link](https://github.com/adobe-research/MakeItTalk)  |
| MakeItTalk paper | [Link](https://arxiv.org/abs/2004.12992)  |
