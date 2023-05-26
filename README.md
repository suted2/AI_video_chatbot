# alpaco_5th_3

## PROJECT Name :  실버 AI 화상 상담 챗봇 



---

### 개요
- 문제 상황
- Project 설명
- 모델 설명
- Dataset
- Reference 


---
### 문제 상황

![문제 배경1](https://github.com/suted2/alpaco_5th_3/blob/13a0d7c32ab6cf61f2a28a00e47e6461ef5faff3/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EB%B0%B0%EA%B2%BD1.png)

🤔 OCED 국가중 가장 낮은 고령층 디지털 숙련도를 가지고 있다. 

*디지털 숙련도란 ? ▶️ 디지털 기기를 사용하여 정보를 얻고 문제를 solving 하는 능력*

> 코로나 19 이후 언택트(Untact)시대에 진입하며 디지털 시대로의 진입이 가속화 되었다. 해당 상황에서</br>
> 노인들의 낮은 디지털 숙련도는 해당 시대의 낮은 적응력을 의미하고 이는 정보의 불균형, 노인 인구의 불만족을 나타낸다. </br>
> 고령화 시대로 진입하고 있는 지금 해당 문제에 대한 해결책이 필요하다.  </br>


  

![문제 배경2](https://github.com/suted2/alpaco_5th_3/blob/13a0d7c32ab6cf61f2a28a00e47e6461ef5faff3/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EB%B0%B0%EA%B2%BD2.png)




![문제 배경3](https://github.com/suted2/alpaco_5th_3/blob/13a0d7c32ab6cf61f2a28a00e47e6461ef5faff3/%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9Cppt/%EB%B0%B0%EA%B2%BD3.png)



### PROJECT 설명 

*추후 작성 예정 * 
---

### ABOUT MODEL

---

### Dataset
---
+  AI hub / 한국어 음성 (입 모양 ) 영상 + 음성 / [You can Download Here!](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=538)

+ 


---
## TRAIN 

| Env |CPU | GPU | RAM | OS 
|------------|------|-------|--------|-----------|
| Local |i5- 13500k | RTX-3070Ti | 32G| Window11 |
| AWS |  AMD-EPYC-7R32 | RTX-3090| 12G| Ubuntu |
| kaggle | intel Xeon | P100 | 12G | Ubuntu | 
| Colab + | intel Xeon | A100 | 80G | Ubuntu |



---
__Wav2LIP__

<img src="https://user-images.githubusercontent.com/101646531/235811260-f4def410-14ec-406f-a0c4-c68fb31c0fed.gif" width="300" height="200"/> <img src="https://user-images.githubusercontent.com/101646531/235811264-d298537e-8a68-42a9-b8f0-f5395f2bfb7a.gif" width="300" height="200"/>


MakeItTalk


![배성재_makeittalk_kor](https://user-images.githubusercontent.com/121469546/235813171-b01d5e9c-4f2f-4c81-93d9-0818c5b4bf73.gif)
![배성재_makeittalk_eng](https://user-images.githubusercontent.com/121469546/235813155-73a2b65a-10da-4e75-afa7-9f9859a0f5a3.gif)





## ReFerence 


|Git|paper|
|---|-----|
|[wav_2lip](https://github.com/Rudrabha/Wav2Lip)| [paper](https://arxiv.org/pdf/2008.10010v1.pdf)|
|[MakeItTalk](https://github.com/yzhou359/MakeItTalk) | [paper](https://arxiv.org/pdf/2004.12992v3.pdf)|
|[ESPNET(JETS)](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2203.16852) |
