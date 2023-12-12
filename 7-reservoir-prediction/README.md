#7 

분석 주제 및 목표: 지자체 배수지 유출유량 분석을 통한 용수수요 예측 파이프라인 구성


핵심 아이디어: 논산배수지의 1주일 전까지의 데이터를 이용해 1주일 전의 데이터를 예측하는 모델을 Docker와 FastAPI를 이용해 Serving하고 Monitoring하는
파이프라인 구축

<img src = 'images/architectur.png' alt = 'Drawing'/>

## 디렉토리 구조

7-reservoir-prediction

- app
    - api.py
    - schemas.py
- artifacts
    - model.joblib
    - scaler.joblib
- docker-compose.yml
- Dockerfile
- requirements.txt
- train.py

## 1. FastAPI Serving API 생성

