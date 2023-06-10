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









### Reference
Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020.
