from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
from schemas import Reservoir, Flow
import json
from train import request_json_data
from fastapi.responses import JSONResponse
import requests
# from .monitoring import instrumentator

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
# scaler = load(ROOT_DIR / "artifacts/scaler.joblib")
model = load(ROOT_DIR / "artifacts/model.joblib")


@app.get("/")
def root():
    return "Flow"

@app.get("/transform")
def transform():
    try:
        # JSON 파일을 읽기
        with open("output.json", "r") as json_file:
            # JSON 파일 내용을 파이썬 객체로 로드
            data = json.load(json_file)

        # 읽어온 데이터 출력
        response_data = {"status": "success", "data": data}
        # JSONResponse.content = json.dumps(response_data)
        JSONResponse.content = response_data

    except FileNotFoundError:
        response_data = {"status": "error", "message": "파일이 존재하지 않습니다."}
        JSONResponse.content=response_data
        # JSONResponse.content = json.dumps(response_data)

    return JSONResponse(content=JSONResponse.content, media_type="application/json")

# sample=Reservoir: return값으로 나오는 model의 데이터 형태: 일주일 동안 한시간 간격의 유출유량 데이터 (:29642,168,1) input data역할
# flow: 전체 데이터에서 1시간 전까지의 유출유량 데이터 (29642,)
# sample: request의 데이터 모델.. request에 유출유량 데이터를 넣으면 되나? jsonify된?
@app.post("/predict", response_model=Flow)
def predict(response: Response, sample: Reservoir):
    return {"status": "success"}
    # sample_dict = sample
    print(sample)
    
    prediction_data = model.predict(sample)
    # print(prediction_data)
    # return 123
    # features = np.array([sample_dict[f] for f in feature_names]).reshape(-1, 1)
    # reshaped_features = features.reshape((features.shape[0], 168, 1))
    # features_scaled = scaler.transform(features)
    # prediction = model.predict(reshaped_features)
    # response.headers["X-model-score"] = str(prediction_data)
    # return Flow(prediction_data)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}