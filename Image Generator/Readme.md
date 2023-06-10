# Image Generator

기존의 챗봇에서 화상 시스템을 더해서 실버층에게 조금 더 편한 UI, UX 그리고 실제로 대화하는 느낌을 주고자 만들게 되었습니다. 

그중 가장 중요하다고 느껴지는 영상을 Generate 하기 위한 모델을 선정해야 했습니다. 

### Relate Work 

> Talking Head Generation </br>  
> Talking Face Generation 

해당 두개의 모델을 집중적으로 찾았습니다. 

###  Model Selection

1. [Make It Talk](https://github.com/suted2/AI_video_chatbot/tree/f8b64f0c32ade7882934a771b3180cda89a302f4/Image%20Generator/MakeItTalk)
2. Wave to Lip 
3. RAD-NeRF

모델 select의 기준은 
1. 여러 사람에게 General 하게 적용 가능한가. 
2. 실제 말하는 것과 위화감이 없게 조절 할 수 있나 

두 가지 였습니다. 

## data

[AI HUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=538)
➡️ 해당 링크에서 받을 수 있는 데이터를 공통적으로 사용했습니다. 

데이터는 한국인 전용 입모양 맞춤 데이터set입니다. 
구성은 mp4 와, 소리만 있는 wav파일, 그리고 메타데이터 정보와 직접 말한 대사가 들어있는 json 파일이 존재합니다. 

현재 공통적으로 3개의 모델이 25fps의 영상을 가지고 있기에 전처리 함수를 통해 시현 하였습니다. 



## Suported models

| model | Git | Paper | 
| :-------------: | :---------------: | :---------------: |
| MakeItTalk | [Git](https://github.com/adobe-research/MakeItTalk) | [Paper](https://arxiv.org/abs/2004.12992)  |
| Wav2Lip | [Git](https://github.com/Rudrabha/Wav2Lip) | [Paper](https://arxiv.org/pdf/2008.10010.pdf)  |


