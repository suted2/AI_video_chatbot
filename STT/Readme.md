# Whisper

OpenAI 에서 제공한 다국어 STT 모델(Whisper)을 사용하였습니다.

Whisper 모델은 Large dataset(680k hours)으로 학습되었고, 다국어 Speech-Recognition, Speech-Translation, Language-Identification이 가능합니다.


## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

Whisper 모델은 위와 같은 구조를 가지고 있고, Transformer, Sequence-to-Sequence 모델은 다양한 Task에 활용될 수 있도록 학습되었습니다. 

## Setup

Window에서 Web Inference를 위해 Python 3.8.10, [PyTorch](https://pytorch.org/) 2.0.1에 맞는 환경을 생성하였습니다.

Whisper 모델의 설치는 다음 Command를 이용해서 가능합니다.

    pip install -U openai-whisper

Github를 이용한 설치도 가능합니다.

    pip install git+https://github.com/openai/whisper.git 


구현을 위해서는 [`ffmpeg`](https://ffmpeg.org/)가 Inference하려는 환경에 설치 되어있어야 했습니다.

저희는 로컬 Desktop에 [`ffmpeg`](https://ffmpeg.org/)에서 최신 버전을 다운로드 받고 **C드라이브에 설치**한 후, **환경변수에 ffmpeg를 추가**해야 합니다.

## Available models and languages

Whisper 모델에는 Parameter에 따른 다양한 Size가 존재합니다. 저희는 한국어로 Whisper 모델을 사용하고자 하여, English-only모델이 아닌 기본 모델을 사용하였습니다. 추가적으로, Inference Time, Accuracy를 고려하여 Medium 모델을 사용하기로 결정하였습니다.

|  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:-------------:|:--------------:|
| medium |   769 M    |    `medium`      |     ~5 GB     |      ~2x       |


## Python usage

저희는 간단하게 Whisper Github에서 제공된 코드와 동일하게 Inference할 수 있도록 구성하였습니다.

```python
import whisper

model = whisper.load_model("medium") # Accuracy & Inference time 고려하여 Size 선정
result = model.transcribe("audio.mp3") # STT를 진행할 mp3파일
print(result["text"])
```

## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
