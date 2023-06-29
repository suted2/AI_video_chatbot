## Toxic Check

#### 욕설 감지에서 가장 중요하게 생각한 것은 속도와 정확도 였습니다. 


[해당](https://github.com/sgunderscore/hatescore-korean-hate-speech.git) git을 통해 사용했습니다. 

---

### DATA 


데이터는 SmileGate AI 에서 제공한 게임 욕설 데이터 입니다. 



[Korean Unsmile data](https://github.com/smilegate-ai/korean_unsmile_dataset) 해당 링크에서 확인 가능합니다. 



### 카테고리 

- [여성/가족]
여성성 및 여성의 성역할에 대한 통념을 고착시키는 발언, 여성 차별을 희화화하는 발언, 페미니즘·여성가족부 전반에 대한 악플 등을 포함합니다.

- [남성]
집단으로서의 남성 일반을 비하, 조롱, 희화화하는 발언들입니다.

- [성소수자]
성소수자(레즈비언, 게이, 바이섹슈얼, 트랜스젠더 등)를 배척하는 발언입니다.

- [인종/국적]
특정 인종(흑인, 아시안 등)과 국적(일본인, 아프가니스탄인, 베트남인 등)에 대한 욕설, 고정관념, 조롱을 다룹니다.

- [연령]
특정 세대나 연령을 비하하는 은어의 사용 및 혐오 표현을 분류하였습니다.

- [지역]
특정 지역에 대한 은어 및 혐오 표현을 분류하였습니다.

- [종교]
특정 종교에 대한 혐오 및 종교인 집단에 대한 비난을 분류하였습니다.

- [기타혐오]
위에서 정의한 카테고리 이외의 집단을 대상으로 하는 혐오 표현을 분류하였습니다. (e.g. 장애인, 정부, 기자, 경찰, 차별금지법 반대 등)

- [악플/욕설]
어떤 집단을 향한 혐오 표현인지 지칭할 수는 없지만, 타인 혹은 외모에 대한 비하/욕설이 포함되어 있거나, 불쾌감을 주거나, 악플과 음란성 문장을 분류하였습니다.

- [Clean]
혐오표현, 욕설, 불쾌감, 음란성 내용을 포함하고 있지 않은 일반 문장을 분류하였습니다.


---
#### Usage

```python
>>> from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
>>> model_name = 'sgunderscore/hatescore-korean-hate-speech'
>>> model = BertForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
>>> pipe = TextClassificationPipeline(
        model = model,
        tokenizer = tokenizer,
        device = -1, # gpu: 0
        return_all_scores = True,
        function_to_apply = 'sigmoid')
>>> for result in pipe("착한 중국인은 죽은 중국인이다")[0]:
        print(result)
    
{'label': 'None', 'score': 0.07771512866020203}
{'label': '기타 혐오', 'score': 0.02803093008697033}
{'label': '남성', 'score': 0.013538877479732037}
{'label': '단순 악플', 'score': 0.01559345331043005}
{'label': '성소수자', 'score': 0.014305355027318}
{'label': '여성/가족', 'score': 0.014650419354438782}
{'label': '연령', 'score': 0.014001855626702309}
{'label': '인종/국적', 'score': 0.9227811098098755}
{'label': '종교', 'score': 0.035127196460962296}
{'label': '지역', 'score': 0.02069076895713806}

```
다음과 같은 방식으로 hugging face의 모델을 사용, Bert 기반의 tokenizer를 사용하여 각 카테고리의 sigmoid 값을 통해 구합니다. 


해당 TOXIC check 는 

STT -> 이후 사용됩니다.
