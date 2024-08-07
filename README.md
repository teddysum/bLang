![image](https://github.com/user-attachments/assets/2213be67-64de-4e52-b800-21ac286b467c)

# bLang

bLang은 Teddysum에서 개발한 LLM 활용 도구 입니다.
최신 개발된 LLM 및 bllossom을 쉽게 활용할 수 있도록 해주는 도구입니다.

# 설치

### huggingface, pytorch 등 llm 개발환경 구성
#### docker를 이용한 설치
   ```
   docker pull blang
   ```

image를 이용하여 container를 연다. 

   ```
   docker run -d -it --name ‘container 이름’ --gpus all blang
   ```

#### bLang 라이브러리 설치
pip를 이용한 설치
```
pip install git+https://github.com/teddysum/bLang
```

conda를 이용한 설이


# 활용법

base 모델 목록 확인
```
blang.get_model_list()
```
