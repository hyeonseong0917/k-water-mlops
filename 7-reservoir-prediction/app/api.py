from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
from schemas import Reservoir, Flow
import json
from train import request_json_data, find_ranges, get_outlier, prepare_dataset
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import math
# from .monitoring import instrumentator

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
# instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)
# scaler = load(ROOT_DIR / "artifacts/scaler.joblib")
model = load(ROOT_DIR / "app/artifacts/model.joblib")


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
# total: 37261?
@app.post("/predict",response_model=Flow)
def predict(response: Response, sample: Reservoir):
    sample_list=list(sample.data.values())
    sample_np=np.array(sample_list)
 
    two_dim_array_reshape = np.array([sample_np])

    target_col_values = two_dim_array_reshape.reshape(37262,1)
    train_len = math.ceil(len(target_col_values) * 0.8)  # 29810

    # 8:2로 데이터 분할
    train_df = target_col_values[0:train_len, :]  # 0~29809
    test_df = target_col_values[(train_len-168):, :]  # 24시간 * 7일
    
    train_df = target_col_values[0:train_len, :]  # 0~29809
    test_df = target_col_values[(train_len-168):, :]  # 24시간 * 7일 
    train_min = np.min(train_df)
    train_max = np.max(train_df)
    scaled_train = (train_df-train_min) / (train_max-train_min)  
    scaled_test = (test_df-train_min) / (train_max-train_min)  
    
    # x_train, y_trian - lag 적용
    # append(데이터): list에 데이터를 추가
    x_train = []
    y_train = []
    for i in range(168, train_len):
        x_train.append(scaled_train[(i-168):i, 0])
        y_train.append(scaled_train[i, 0])
    # x_test, y_test - lag 적용
    #len(데이터): 데이터의 길이(데이터프레임은 행의수, 리스트는 데이터의 수)를 출력
    x_test = []
    y_test = scaled_test[168:, :]
    for i in range(168, len(test_df)):
        x_test.append(scaled_test[(i-168):i, 0])
    # 8:2로 데이터 분할
    train_df = target_col_values[0:train_len, :]  # 0~29809
    test_df = target_col_values[(train_len-168):, :]  # 24시간 * 7일 
    train_x_array = np.array(x_train)
    train_y_array = np.array(y_train)
    test_x_array = np.array(x_test)
    # lstm학습하기 위해 reshape

    # reshape(): array의 차원을 변형
    train_x_array = np.reshape(train_x_array, (train_x_array.shape[0], train_x_array.shape[1], 1))  
    test_x_array = np.reshape(test_x_array, (test_x_array.shape[0], test_x_array.shape[1], 1)) 
    
    prediction_data = model.predict(train_x_array)
    rounded_prediction_data = np.round(prediction_data, decimals=4)
    # print(rounded_prediction_data)
    reshaped_array = rounded_prediction_data.reshape(29642)
    
    input_data={'data': reshaped_array}
    print(Flow(**input_data))
    # response.headers["X-model-score"] = json.dumps(prediction_data.tolist())
    # print(response.headers["X-model-score"])
    return Flow(**input_data)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}
