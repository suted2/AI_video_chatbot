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

```


### Reference 

TANG, Jiaxiang, et al. Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition. arXiv preprint arXiv:2211.12368, 2022.
