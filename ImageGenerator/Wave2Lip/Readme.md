# Wave to Lip 


## Data 전처리 

1. 해당 데이터의 경우 25fps로 고정 해주었고 해상도를 720 x 720 으로 맞춰 주었다. 
2. 해당 데이터를  (원하는 폴더명)/main/(원하는 숫자)/ 아래 .mp4와 해당하는 파일의 txt파일을 유지하며 넣어준다.
3. filelists/ 폴더 내에 train.txt, valid.txt 등 원하는 파일 경로가 담겨있는 텍스트 파일을 더한다.  
4. 이후 `python process.py --batch_size <depend on your Ram> --data_root <2번에서 만들었던 main폴더> --preprocessed_root <전처리된 데이터를 두고싶은 위치>`

5. 해당 전처리는 50시간의 영상길이에 3일정도 걸렸습니다. 

## Model 선정 이유 

1. 입모양이라는 대화에 있어서 위화감을 조성하는 가장 큰 비언어적 표시를 target으로 train하기 위해 고르게 되었습니다. 
2. 한국어 이외의 5개 언어로 pre-train되어있어 한국어 inference 시 이빨이 안나오거나, 입모양이 맞지 않는 문제가 발생합니다. 
3. 따라서 fine-tuning을 진행 하려고 합니다 .



## How To fine-tuning own your data 

> Requirements의 경우 본 논문에 제시되었던 환경에 큰 문제가 존재하지 않았습니다. (Window 11 기준) 
> librosa 모듈의 경우 현재 (23.06.10 ) 기준 0.10.xx가 최신 버전이지만 0.9.xx로 낮춰야 문제 없이 실행됩니다. 
> log 의 경우 wandb를 통해 확인 할 수 있도록 연결 되어있습니다. 개인 wandb에 login 이후 확인 가능합니다. 

1. Train syncnet
`python color_syncnet_train.py --data_root ./my_preprocessed/ --checkpoint_dir checkpoints/ --checkpoint_path checkpoints/lipsync_expert.pth`

2. Train wav2lip 

`python wav2lip_train.py --data_root ./my_preprocessed/ --checkpoint_dir checkpoints/ --syncnet_checkpoint_path <your syncnet.pth>`

3. If you want hq_train 
> 위의 train은 입모양에 집중하기에 화질 자체에는 생성 문제가 있습니다 따라서 추가적인 discriminator를 더하여 해결 가능합니다 .

`%run hq_wav2lip_train.py --data_root <your> --checkpoint_dir checkpoints/ --syncnet_checkpoint_path <your syncnet.pth> --checkpoint_path <wav2lip.pth (이전 트레인에서 이어서 할 경우> --disc_checkpoint_path <화질 관련 discriminator.pth>`


---

## 왜 Fine-Tuning 이 필요한가 ? 

1. 영어 및 4개의 다국어로 pretrain 된 본 논문의 weight를 가지고 inference 시 한국어를 하면 이빨이 사라지는 모습이 보인다. 
![image](https://github.com/suted2/AI_video_chatbot/assets/101646531/9e92dd46-90a9-43af-a932-5523d9690b16)

2. 한국어 입모양이 맞지 않다. <br>
![영어 입모양](https://github.com/suted2/AI_video_chatbot/assets/101646531/f93e3a2e-ea65-45b4-bdde-a2800c64fea8)
![한국어 적용 입모양](https://github.com/suted2/AI_video_chatbot/assets/101646531/70d69f34-8771-4f85-8139-1dc992122f60)




### 개인 적인 Train 결과 

1. syncnet ( Discriminator ) <br>
![image](https://github.com/suted2/AI_video_chatbot/assets/101646531/f2e6040c-aa1c-45b7-bc16-4617ac6a24ef)

논문에서 요구하는 수치인 0.25 loss 까지는 수렴하지 못했지만, 0.44 까지 수렴 Train 시간이 110 시간이 넘어가서 중지함. 



2. Generator <br>
![image](image.png)

논문에서 요구하는 수치인 0.2 loss를 달성하여 수렴하게 되었다. 




### Reference
Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020.
