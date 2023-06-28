
# 프로젝트 설명

## 프로젝트 구성도

![Alt text](png/image-1.png)
- input으로 질문이 들어오면, 1차적으로 답변 DB에서 Poly Encoder로 k개의 답변 후보 선정
- 2차로 Cross Encoder를 통해 최종 답변 선정해 output

## Retrieval-Based
- 생성형 챗봇인 Chat GPT는 Hallucination현상 등으로 인해 정확하지 않은 답변을 도출하는 경우 있음
- 정확한 답변만을 해야하는 상담 챗봇 특성으로 인해 Retrieval System Chatbot을 선정
- 또한 생성형에 비해 답변 속도가 비교적 빠름

## Poly Encoder
 ![Alt text](png/image-2.png)
- 기존 Bi Encoder
## Cross Encoder
![Alt text](png/image-3.png)
## Problem

## Solving

---
# 학습 부분


## Requirements

- requirements.txt 참조


## 학습 위한 Bert Model Setup

- BERT 관련 다른 원하는 모델있으면 사용가능
1. [BERT model] 
   - huggingface 'kykim/bertshared-kor-base' (url : https://huggingface.co/kykim/bertshared-kor-base/tree/main) 에서 [pytorch_model.bin, config.json, tokenizer_config.json, vocab.txt] 다운로드 후 models/bert에 저장

2. [RoBERTa model] 
   - huggingface 'klue/roberta-large' (url : https://huggingface.co/klue/roberta-large/tree/main) 에서 [pytorch_model.bin, config.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.txt] 다운로드 후 models/roberta에 저장

## 데이터셋


1. 원본 데이터 : AIhub 민원(콜센터) 질의-응답 데이터(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=98) 라벨링 데이터 중 다산콜센터 데이터 (json) 사용

2. 다산콜센터 상담 데이터 중 일반행정, 상하수도, 코로나 관련 주제 데이터 사용

3. 각 주제별로 상담에 유용하다고 생각되는 [질문 -답]쌍 250개 이상 추출, 생성

4. 데이터 학습을 위한 데이터 형식 예시

|       **index**       |   **q1**  |   **q2**  |  **q3**   |  **q4**  |  **response**   |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|   0    |   코로나 자가격리시 일을 못하는데 어떻게 하나요?    |   자가격리시 일을 못하는데 어떻게 하나요?    |   자가격리로 일을 못하는데 방법이 있나요 ?    |   코로나 자가격리때문에 일을 못하는데 어떻게 하나요?     |   정부에서 별도 지원금을 지급하고 있습니다    |
| 1  |  코로나 자가격리 지원금을 위한 필요서류가 있나요?    |   코로나 자가격리 지원금을 위해 준비해야할 필요서류가 있나요?    |   코로나 자가격리 지원금 신청할때 준비할 필요서류가 있나요?    |   코로나 자가격리 지원금 신청할때 서류가 있나요?     |   신청서와 신분증 사분, 자가격리이행 확약서 등입니다.    |
- 챗봇에서 사람들마다 질문하는 방식이 다를 수 있음
   - 하나의 답변에 대해 의미는 같으나 다양한 문장 구조를 가진 질문을 총 4개의 질문을 준비
- pickle 확장자 파일로 준비

## 학습 코드 예시
1. Train **Poly-Encoder**
   ```shell
   python utils/run.py \
   --model_type bert \
   --bert_model models/bert/ \
   --output_dir result/train1 \
   --train_dir datasets/ \
   --train_file dasan_train_data.pickle \
   --valid_file dasan_train_data.pickle \
   --use_pretrain \
   --architecture poly \
   --poly_m 16 \
   --train_batch_size 2 \
   --eval_batch_size 2 \
   --max_contexts_length 256 \
   --max_response_length 64 \
   --num_train_epochs 1000
   ```
2. Train **Cross-Encoder**
   ```shell
   python utils/run.py \
   --model_type bert \
   --bert_model models/bert/ \
   --output_dir result/train2 \
   --train_dir datasets/ \
   --train_file dasan_train_data.pickle \
   --valid_file dasan_train_data.pickle \
   --use_pretrain \
   --architecture cross \
   --train_batch_size 2 \
   --eval_batch_size 2 \
   --max_contexts_length 256 \
   --max_response_length 64 \
   --num_train_epochs 1000
   ```

## Inference

1. 답변 후보 text들의 embedding을 미리 계산하기 위하여 text2emb.py 실행
   ```shell
   python utils/text_2_emb.py \
   --model_type bert \
   --bert_model models/bert \
   --text_path /path/to/카테고리별답변들.txt \
   --output_dir /path/to/카테고리별embedding.pickle \
   --gpu 1
   ```
   - 카테고리별답변들.txt 파일은 한 줄마다 학습시킨 답변들 하나씩 작성
   - output인 카테고리별embedding.pickle의 예시는 우리 datasets/{category}_with_text.pickle 를 통해서 확인 가능
2. 이후 inference.py를 통하여 모델 생성하여 챗봇 모델 생성하여 [질문->답변] 실험 가능
   - 자세한 실행 코드는 examples 폴더의 how_2_inference.ipynb 파일 참조



## 학습환경


|       모델       |       환경       |   **CPU**  |   **RAM**  |  **GPU**   |  **OS**  |  **Training Time**   |
| :---------------: | :---------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|    Poly Encoder    |    AWS    |   AMD-EPYC   |   16G    |   A10(24G)    |   Ubuntu     |   24hr    |
|    Cross Encoder    |    Colab pro+    |   Intel-xeon   |   80G    |   A100(40G)    |   Ubuntu     |   21hr    |


---
# Bi-Encoder, Poly-Encoder, and Cross-Encoder for Response Selection Tasks

- This repository is an unofficial re-implementation of [Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://arxiv.org/abs/1905.01969v4).

- Special thanks to sfzhou5678! Some of the data preprocessing (dataset.py) and training loop code is adapted from his [github repo](https://github.com/sfzhou5678/PolyEncoder). However, the model architecture and data representation in that repository do not follow the paper exactly, thus leading to worse performance. I re-implement the model for Bi-Encoder and Poly-Encoder in encoder.py. In addition, the model and data processing pipeline of cross encoder are also implemented.

- Most of the training code in run.py is adpated from [examples](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py?fbclid=IwAR3BlKIJYak659a6X12gsYOMs1JJPtnsdFUmn93CovwTJ5VXQZX1TK78yGo#L102) in the [huggingface](https://github.com/huggingface/transformers) repository.

- The most important architectural difference between this implementation and the original paper is that only one bert encoder is used (instead of two separate ones). Please refer to this [issue](https://github.com/chijames/Poly-Encoder/issues/4#issue-728612873) for details. However, this should not affect the performance much.

- This repository does not implement all details as in the original paper, for example, learning rate decay by 0.4 when plateau. Also due to limited computing resources, I cannot use the exact parameter settings such as batch size or context length as in the original paper. In addition, a much smaller bert model is used. Feel free to tune them or use larger models if you have more computing resources.
