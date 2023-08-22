
# PROJECT Name :  실버 AI 화상 상담 챗봇
</br></br></br>

## 프로젝트 기간 📆

|날짜|업무 내용|
|:--:|:--:|
|2023.04.24 ~ 2023.04.28|사전 기획(프로젝트 기획, 주제 선정,  자료 조사)|
|2023.04.29 ~ 2023.05.08|Data 수집,정제(lip sync 영상 수집, 정제, TTS Data 녹음, 전처리)|
|2023.05.08 ~ 2023.06.13|Image Generation Modeling(Make It Talk, Wave To Lip, Rad-NeRF)|
|2023.05.15 ~ 2023.05.23|TTS Modeling(JETS)|
|2023.05.24 ~ 2023.06.13|ChatBot system(Poly-encoder)|

</br></br>

## 구성원 🤸🏻‍♀️
*알파코 5기 3조* 

|구성원|깃허브 주소|분담 역할|
|:---:|:--:|:--:|
|노아윤|[Git](https://github.com/ayun3738)|프로젝트 일정 관리, MIT fine-tuning, 데이터 전처리, <br> Web Ux 구현|
|김도현|[Git](https://github.com/doh0106)|TTS Modeling(Jests-based), 데이터 전처리, Poly-Encoder train, <br> aws 환경 구축, Web Ux 구현|
|송기훈|[Git](https://github.com/Kihoon9498)|MIT fine-tuning, 데이터 전처리, whisper pipeline, <br> aws 환경 구축, Web UI, Ux 구현, 웹 서비스 배포|
|황민규|[GIt](https://github.com/suted2)|W2L fine-tuning, Rad-NeRF fine-tuning 데이터 전처리, <br> TOXIC pipeline, TTS 데이터 생성, Web Ux 구현|

</br>  

---

## Enviroment

| Env |CPU | GPU | RAM | OS 
|:--:|:--:|:--:|:--:|:--:|
| Local |i5- 13500k | RTX-3070Ti | 32G| Window11 |
| AWS |  AMD-EPYC-7R32 | RTX-3090| 12G| Ubuntu |
| kaggle | intel Xeon | P100 | 12G | Ubuntu | 
| Colab + | intel Xeon | A100 | 80G | Ubuntu |


---

### Contents Table
- [문제 상황](#문제-상황)
- [기대 효과](#)
- [Project 설명](#PROJECT-설명)  
- [모델 설명](#about-model)
- [Dataset](#Dataset)
- [Reference](#Reference) 

---
### 문제 상황

![문제 배경1](https://github.com/suted2/alpaco_5th_3/blob/13a0d7c32ab6cf61f2a28a00e47e6461ef5faff3/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EB%B0%B0%EA%B2%BD1.png)

🤔 OCED 국가중 가장 낮은 고령층 디지털 숙련도를 가지고 있다. 

*디지털 숙련도란 ? ▶️ 디지털 기기를 사용하여 정보를 얻고 문제를 solving 하는 능력*

> 코로나 19 이후 언택트(Untact)시대에 진입하며 디지털 시대로의 진입이 가속화 되었다. </br>
> 노인들의 낮은 디지털 숙련도는 디지털 시대의 낮은 적응력을 의미하고 이는 정보의 불균형, 노인 인구의 불만족을 나타낸다. </br>
> 고령화 시대로 진입하고 있는 지금 해당 문제에 대한 해결책이 필요하다.  </br>

</br></br>
  
우리는 그렇다면 해당 문제를 해결하기 위해 **어떤 문제를 노인들이 어려워하는지 알 필요가 있다 .** 🤔 

![문제 배경2](https://github.com/suted2/alpaco_5th_3/blob/13a0d7c32ab6cf61f2a28a00e47e6461ef5faff3/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EB%B0%B0%EA%B2%BD2.png)


- 해당 표를 통해 알 수 있는 것은 노인들이 사용방법을 모르거나 어려워서 디지털 기기를 사용하지 못한다는 것을 알 수 있다.</br>


</br></br>

그렇다면 고령층에게 가장 **필요한** 서비스는 어느 분야일까 ❔
 
![문제 배경3](https://github.com/suted2/alpaco_5th_3/blob/13a0d7c32ab6cf61f2a28a00e47e6461ef5faff3/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EB%B0%B0%EA%B2%BD3.png)

</br>
해당 표에 나오듯 공공 서비스에 대한 정보 / 역량이 가장 중요하다는 사실을 확인 할 수 있다. . </br></br></br></br>


 ✔️ **현재 시중의 챗봇 , 디지털 시스템의 문제를 확인해 보자**


![시중 챗봇의 문제1](https://github.com/suted2/alpaco_5th_3/blob/3586b1cb84ad6cea4915d0f63c0bc3a36df55693/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EC%8B%A4%EC%A0%9C%20%ED%8C%BB%EB%B4%87%EC%98%88%EC%8B%9C.png)
</br>
1. 고령층의 디지털 기기의 가장 큰 문제는 **많은 글자**이다.</br>

> 카이스트 연구에 따르면 고령층은 청년층에 비해 글자를 인지하고, 이해하는데 걸리는 시간이 30% 이상 느리다. </br>


![시중 챗봇의 문제2](https://github.com/suted2/alpaco_5th_3/blob/3586b1cb84ad6cea4915d0f63c0bc3a36df55693/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EC%8B%A4%EC%A0%9C%EC%98%88%EC%8B%9C%202.png)
</br>

2. 고령층이 디지털 기기에서 마주할 다른 문제는 **복잡하고 긴 절차**이다. </br>

> 상담원과의 전화 연결은 모든 질문에 대한 답이 한번에 오는 반면 </br>
> 디지털 기기를 통한 상담은 긴 절차와 결과창까지 확인하는 시간이 오래 걸리며 복잡하다. </br>

</br></br>

**결론적으로 만들게 될 모델은 실버 + AI상담원 + 챗봇**이다. 

실버 `고령층을 위한 쉽고 편리한` + AI상담원`실제 사람과 대화하는 경험을 통해 거부감을 줄인` </br>
챗봇`공공서비스에 대한 정보를 제공하는` 시스템을 만드는 PROJECT이다. 



</br></br>

### 기대효과 


+ 노인들의 만족도 상승
  > 노인들은 디지털 시대에 적극적으로 참가하며, 원하는 정보를 얻고 참가하기에 사회 전반적인 시스템에 자신감이 생기고 만족한다. </br>
+ 24시간 상황 대처 가능 
  > 24시간 고령층의 민원에 대응이 가능하며, 응급상황, 다양한 민원에 대응이 가능하다. 
+ 예산 감소
  > 현재 많은 예산과 인원이 고령층을 위한 정책 시행, 홍보에 소모되고 있다. 고령층이 능동적으로 정보를 찾고 받을 수 있는 능력이 있다면 해당 인원 돈을 단축할 수 있다.


</br></br>

### PROJECT 설명 

![총 프로세스](https://github.com/suted2/AI_video_chatbot/assets/101646531/5e456e6f-093a-4336-9236-74824bc61957)


</br></br>

❗STT 를 통해 들어온 목소리를 텍스트로 바꾼다. 해당 모델에 대한 자세한 내용은 ➡️ [여기](https://github.com/suted2/AI_video_chatbot/tree/main/STT) 확인 가능합니다. <br>
❗TOXIC 를 통해 들어온 목소리를 텍스트로 바꾼다. 해당 모델에 대한 자세한 내용은 ➡️ [여기](https://github.com/suted2/AI_video_chatbot/tree/main/TOXIC) 확인 가능합니다. <br>
❗Chat Bot을 통해 원하는 답변을 얻어 답변하는 Model 일련의 과정은 ➡️ [여기](https://github.com/suted2/AI_video_chatbot/tree/main/Chatbot) 확인 가능합니다. <br>
❗TEXT TO SPEECH 를 통해 텍스트를 목소리로 바꾼다. 해당 모델에 대한 자세한 내용은 ➡️ [여기](https://github.com/suted2/AI_video_chatbot/tree/main/TTS) 확인 가능합니다.  
❗Image Generation 을 통해 목소리와 base model 사진을 넣어 **영상을 생성하는** 내용은 ➡️ [여기](https://github.com/suted2/AI_video_chatbot/tree/main/Image%20Generator) 확인 가능합니다.  



---

### ABOUT MODEL


*모델들의 예시를 확인할 수 있는 곳 입니다.*

__Wav2LIP__

<img src="https://user-images.githubusercontent.com/101646531/235811260-f4def410-14ec-406f-a0c4-c68fb31c0fed.gif" width="300" height="200"/> <img src="https://user-images.githubusercontent.com/101646531/235811264-d298537e-8a68-42a9-b8f0-f5395f2bfb7a.gif" width="300" height="200"/>


MakeItTalk


![배성재_makeittalk_kor](https://user-images.githubusercontent.com/121469546/235813171-b01d5e9c-4f2f-4c81-93d9-0818c5b4bf73.gif)
![배성재_makeittalk_eng](https://user-images.githubusercontent.com/121469546/235813155-73a2b65a-10da-4e75-afa7-9f9859a0f5a3.gif)


__Rad-NeRF__



<img src="https://github.com/suted2/AI_video_chatbot/blob/7a5d4837f9e82e8bacfb762c9aa3f0fd293faea9/Image%20Generator/assets/RadNerf.gif" width="500" height="400"/>

__SadTalker__


<img src="https://github.com/suted2/AI_video_chatbot/blob/dbf0733fe0cc80634c0aef0a8145c62829d75059/Image%20Generator/assets/KakaoTalk_20230626_174102381.gif" width="500" height="400"/>



---

### Dataset
---
+  AI hub / 한국어 음성 (입 모양 ) 영상 + 음성 / [You can Download Here!](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=538)
> 해당 데이터에 대한 자세한 설명은 ➡️ [여기](https://github.com/suted2/alpaco_5th_3/tree/main/Image%20Generator) 확인 가능합니다.


+ 직접 녹음한 / 한국어 남성 독백 wav파일 / `if you want some data contact me by Email!`







---

#### 적용할 수 있는 분야 


![Future work](https://github.com/suted2/AI_video_chatbot/assets/101646531/bc438ba5-1f2f-4174-a618-cd409bdac1f1)

 해당 프로젝트는 AI 화상 상담원을 기준으로 작성되었다.  하지만 여러 분야에 걸처 사용될 수 있다고 생각된다. 

 1. 검색 엔진, 많은 페이지.
    - 실상 많은 고령층이 사용하는 검색 엔진은 네이버, 구글일 것이다. 이 중 사진, 친근성이라는 이유는 네이버가 클 것이다.
    - 해당 검색을 통하면 광고, 너무 많은 URL 등으로 실제로 유용한 정보 탐색에 큰 어려움을 겪고 있다.
    - 검색 엔진에 AI 화상 상담원을 사용하여 실버층의 검색을 도울 수 있다.


  2. 모바일 금융
     - 많은 금융 혜택, 프로세스가 모바일로 넘어가고 있다.
     - 하지만 고령층은 불신과, 과정의 어려움으로 실제로 사용하지 못하고 있다.
    
  3. 현장 키오스크로 확대
     - `키오스크가 가장 노인들이 두려워하는 방식이다.` 라는 말처럼 키오스크의 사용방법과 절차는 고령층에게 매우 어렵다. 
     - 키오스크에 적용 처리한다. 



---


## ReFerence 


|Reference|Git|paper_link|
|:--:|:--:|:--:|
|Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020.|[Wav_2Lip](https://github.com/Rudrabha/Wav2Lip)| [paper](https://arxiv.org/pdf/2008.10010v1.pdf)|
|Zhou, Yang, et al. "Makelttalk: speaker-aware talking-head animation." ACM Transactions On Graphics (TOG) 39.6 (2020): 1-15.|[MakeItTalk](https://github.com/yzhou359/MakeItTalk) | [paper](https://arxiv.org/pdf/2004.12992v3.pdf)|
|[end-to-end speech processing toolkit](https://espnet.github.io/espnet/)|[ESPNET(JETS)](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2203.16852) |
|TANG, Jiaxiang, et al. Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition. arXiv preprint arXiv:2211.12368, 2022.|[Rad-NeRF](https://github.com/ashawkey/RAD-NeRF)| [paper](https://arxiv.org/abs/2211.12368)
|LIM, Dan; JUNG, Sunghee; KIM, Eesung. JETS: Jointly training FastSpeech2 and HiFi-GAN for end to end text to speech. arXiv preprint arXiv:2203.16852, 2022.|[JETS](https://github.com/imdanboy/jets)| [paper](https://arxiv.org/pdf/2203.16852v2.pdf)
|HUMEAU, Samuel, et al. Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring. arXiv preprint arXiv:1905.01969, 2019.|[Poly-Encoder](https://github.com/chijames/Poly-Encoder)| [paper](https://arxiv.org/pdf/1905.01969v4.pdf)
