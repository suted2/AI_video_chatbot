<div align="center">

<img src='https://user-images.githubusercontent.com/4397546/229094115-862c747e-7397-4b54-ba4a-bd368bfe2e0f.png' width='500px'/>


<!--<h2> 😭 SadTalker： <span style="font-size:12px">Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation </span> </h2> -->

  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 


![sadtalker](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)

<b>TL;DR: &nbsp;&nbsp;&nbsp;&nbsp; single image 🙎‍♂️  &nbsp;&nbsp;&nbsp;&nbsp;+  &nbsp;&nbsp;&nbsp;&nbsp; audio 🎤  &nbsp;&nbsp;&nbsp;&nbsp; =  &nbsp;&nbsp;&nbsp;&nbsp; talking head video 🎞.</b>

<br>

</div>

## 📝 0. Paper

### Summary

ExpNet(얼굴) + 

### Main Pipeline
![main pipeline](docs\main_pipeline.PNG)


1. image input을 coefficients of 3DMM(3D Morphable Model)로 첫 이미지의 facial expression($\beta_{0}$), head pose($\rho_{0}$)를 생성한다.
2. audio feature input($\alpha_{\{1...n\}}$)와 $\beta_{0}$를 ExpNet에 통과시켜 눈깜빡임, 입술모양, 얼굴 표정 개선에 관한 연속값을 추출한다.($\beta_{\{1...n\}}$)
3. $\alpha_{\{1...n\}}$와 $\rho_{0}$를 PoseVAE에 통과시켜 Style을 입힌 연속값을 추출한다.($\rho_{\{1...n\}}$)
4. $\beta_{\{0...n\}}$와 $\rho_{\{0...n\}}$을 3D-Aware Face Render 모듈을 통해 연속적인 Frame을 생성한다.


## ⚙️ 1. Installation

### Windows:

1. [Python 3.10.6](https://www.python.org/downloads/windows/) 설치, python 환경변수 설정 체크하기.
2. [git](https://git-scm.com/download/win) 설치
3. `ffmpeg` 설치, [여기](https://www.wikihow.com/Install-FFmpeg-on-Windows)를 통해 설치할 것. (pip install ffmpeg 안될 수 있음).
4. command창에 `git clone https://github.com/Winfredy/SadTalker.git` 입력.
5. [여기서](#📥-2-download-trained-models) `checkpoint`랑 `gfpgan`  다운로드.

## 📥 2. Download Trained Models

**Google Driver**: [main checkpoints](https://drive.google.com/file/d/1gwWh45pF7aelNP_P78uDJL8Sycep-K7j/view?usp=sharing), [gfpgan](https://drive.google.com/file/d/19AIBsmfcHW6BRJmeqSFlG5fL445Xmsyi?usp=sharing)




Model explains:

| Model | Description
| :--- | :----------
|checkpoints/mapping_00229-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/mapping_00109-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/SadTalker_V0.0.2_256.safetensors | packaged sadtalker checkpoints of old version, 256 face render.
|checkpoints/SadTalker_V0.0.2_512.safetensors | packaged sadtalker checkpoints of old version, 512 face render.
|gfpgan/weights | Face detection and enhanced models used in `facexlib` and `gfpgan`.


## 🔮 3. Quick Start

1. conda 가상환경 설치
2. git clone
3. requirements 설치( 230628 기준 requirements)
4. torch cuda11.7 설치( 3070 Ti 기준)

```command
conda create -n sad-talker python=3.10
git clone https://github.com/Winfredy/SadTalker.git
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
5. Inference

```bash
# driven_audio   : 만들 영상의 input wav      (default : ./examples/driven_audio/bus_chinese.wav)
# source_image   : 만들 영상의 image or video (default : ./examples/source_image/full_body_1.png)
# result_dir     : output 폴더               (default : ./results)
# still          : 얼굴만 crop하는 옵션
# enhancer       : 얼굴 더 자연스럽게 후처리   (default : None)
# checkpoint_dir : 모델 checkpoint 폴더       (default : ./checkpoints) 
# size           : facerender size           (default : 256)
```
다른 옵션들은 inference.py에 명시되어 있습니다.

```bash
python inference.py --driven_audio 0_0_00004.wav \ 
--source_image UJaeSuck1.mp4 \
--result_dir ./results \
--still \
--enhancer gfpgan \
--checkpoint_dir SadTalker \
--size 512

```

## 🛎 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2022sadtalker,
  title={SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation},
  author={Zhang, Wenxuan and Cun, Xiaodong and Wang, Xuan and Zhang, Yong and Shen, Xi and Guo, Yu and Shan, Ying and Wang, Fei},
  journal={arXiv preprint arXiv:2211.12194},
  year={2022}
}
```


