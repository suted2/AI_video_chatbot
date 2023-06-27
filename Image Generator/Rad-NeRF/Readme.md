## Rad-NeRF : Real-time Neural Talking Portrait Synthesis

### AD-NeRF ( Audio Driven -Nerf)에서 발전된 모델입니다. 

### R(eal time) + AD-NeRF를 해서 Rad-NeRF라는 표현이 됩니다. 


[해당 Git hub 를 바탕으로 진행했습니다.](https://github.com/ashawkey/RAD-NeRF)




#### Requirements 

2023-06-27 기준 AWS/ Window11 / python 3.10.6 
`pip install -r requirements` 했을 경우 어떤 문제없이 작동되었습니다. 


설치 
```python
# 우분투 환경에서는 portaudio 를 먼저 깔아야  pyaudio 가 작동됩니다. 
sudo apt install portaudio19-dev

pip install -r requirements.txt

```


#### Data Preprocessing 

```python
## install pytorch3d 메타 research 의 3D 컴퓨터 비전을 위한 툴 
pip install "git+https://github.com/facebookresearch/pytorch3d.git" 

## prepare face-parsing model
# AD NeRF( audio driven NerF) 에서 제공하는 체크 포인트를 저장합니다. 
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth

## prepare basel face model
# 1. download `01_MorphableModel.mat` from https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads and put it under `data_utils/face_tracking/3DMM/`

## 1번 mat 파일은 기존의 얼굴 평균 파일 들입니다. 해당 url로 들어가서 다움 가능합니다. ( 받은 파일들을 압축 해제 하면 존재합니다. )


# 2. download other necessary files from AD-NeRF's repository:
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O data_utils/face_tracking/3DMM/exp_info.npy
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O data_utils/face_tracking/3DMM/keys_info.npy
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O data_utils/face_tracking/3DMM/sub_mesh.obj
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O data_utils/face_tracking/3DMM/topology_info.npy
# 3. run convert_BFM.py
cd data_utils/face_tracking
python convert_BFM.py
cd ../..

## prepare ASR model
# if you want to use DeepSpeech as AD-NeRF, you should install tensorflow 1.15 manually.
# else, we also support Wav2Vec in PyTorch.

## 해당 Train 은 전부 wav2vec 기준으로 진행했습니다. 
## hugging face에 존재하는 한국어 Train wav2vec을 사용했습니다. 

```

#### ASR MODEL 

해당 과정에서 mp4에서 오디오 파일을 asr model을 통해서 변환합니다. 
기존 모델은 스페인어 프리트레인 모델을 사용했습니다. 한국어로 진행하기 위해 
'https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean'에 존재하는 
kresnik/wav2vec2-large-xlsr-korean 모델을 썼습니다. 

해당 모델을 사용할 경우 embedding vector의 차원이 달라 코드 내에서 수정이 필요합니다. 



### Train Your own video 

- 경로는 data/<원하는 ID>/<원하는 ID>.mp4 에 넣어주시면 됩니다. 

비디오는 반드시 **25 프레임**으로 구성되어있고 모든 프레임에 말하는 사람의 얼굴이 있어야 합니다.  

영상의 길이는 **1분~5분**으로 구성해야 오류가 나지 않습니다. 

영상의 해상도는 **512*512** 에 맞춰야 합니다. 


Run script (may take hours dependending on the video length)

```python
# run all steps
python data_utils/process.py data/<ID>/<ID>.mp4

# if you want to run a specific step 
python data_utils/process.py data/<ID>/<ID>.mp4 --task 1 # extract audio wave

##


```



### Reference 

TANG, Jiaxiang, et al. Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition. arXiv preprint arXiv:2211.12368, 2022.
