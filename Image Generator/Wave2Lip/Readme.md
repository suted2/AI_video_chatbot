# Wave to Lip 


## Data 전처리 

1. 해당 데이터의 경우 25fps로 고정 해주었고 해상도를 720 x 720 으로 맞춰 주었다. 
2. 해당 데이터를  (원하는 폴더명)/main/(원하는 숫자)/ 아래 .mp4와 해당하는 파일의 txt파일을 유지하며 넣어준다.
3. filelists/ 폴더 내에 train.txt, valid.txt 등 원하는 파일 경로가 담겨있는 텍스트 파일을 더한다.  
4. 이후 `python process.py --batch_size <depend on your Ram> --data_root <2번에서 만들었던 main폴더> --preprocessed_root <전처리된 데이터를 두고싶은 위치>` 
5. 

## Model 선정 이유 










### Reference
Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020.
