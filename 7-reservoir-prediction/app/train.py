# warning 제거

# environ(): 환경변수 설정
# 'TF_CPP_MIN_LOG_LEVEL': 텐서플로에서 ERROR, INFO, WARNING 로그를 출력하지 않는 방법
# 0: 모든 메세지 출력, 1:info메세지만 미출력, 2:info, warning메세지 미출력 , 3: info, warning, error 메세지 미출력
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')
# 패키지 설치
import math  
import pandas as pd  
import numpy as np 
from datetime import datetime  
import logging
from keras.models import Sequential 
from keras.layers import LSTM, Dense  
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from pathlib import Path
import joblib 
from joblib import dump
import json

logger = logging.getLogger(__name__)
request_json_data=[]

def find_ranges(idx):
    # 인덱스를 구간으로 나누어 튜플로 저장
    ranges = []
    start = None
    for num in idx:
        if start is None:
            start = num
        elif num != (idx[idx.index(num)-1] + 1):
            ranges.append((start, idx[idx.index(num)-1]))
            start = num
    if start is not None:
        ranges.append((start, idx[-1]))
        
    # 연속/비연속 분류   
    continuous = []
    discontinuous = []
    for i in range(len(ranges)):
        if ranges[i][0] == ranges[i][1]:
            discontinuous.append(ranges[i][0])
        else:
            continuous.append(ranges[i])
    return continuous, discontinuous

def get_outlier(df, column, weight = 1.5):
    quantile_25 = np.nanpercentile(df[column].values, 25)
    quantile_75 = np.nanpercentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    
    return outlier_idx

def prepare_dataset(random_seed=1):
    data =  pd.read_csv('kwater_recipe07_dataset.csv')
    # print(data.head(30))
    pattern = '[a-zA-Z]|[ㄱ-ㅎㅏ-ㅣ가-힣]'
    for i in data.columns[2:]:
        matches = data[f'{i}'].str.contains(pattern)
        character_raw = matches[matches==True].index
        # print(i, data.loc[character_raw, i].unique())
    # character로 된 데이터 결측(NA)처리
    for i in data.columns[2:]:
        matches = data[f'{i}'].str.contains(pattern)
        character_raw = matches[matches==True].index
        data.loc[character_raw, i] = np.NaN
    for i in data.columns[2:]:  # 배수지 컬럼
        data[f'{i}'] = pd.to_numeric(data[f'{i}'])   
    data['연월일시'] = data['일자'] + ' ' + data['시간']
    data['연월일시'] = pd.to_datetime(data['연월일시'])
    data = data.iloc[13426:, :].reset_index(drop=True)
    data = data.replace(0, np.NaN)

    # 인덱스 분류

    # 인덱스를 리스트에 저장
    idx = []
    con = (data['논산배수지'].isnull() == True)
    for i in data['논산배수지'][con].index:
        idx.append(i)
    
    continuous, discontinuous = find_ranges(idx)
    # NA가 연속으로 나오는 구간 처리
    for i in range(len(continuous)):
        idx1 = continuous[i][0]-1
        idx2 = continuous[i][1]+1
        data['논산배수지'].iloc[continuous[i][0]:idx2] = data['논산배수지'].iloc[idx1]

    # 연속되는 부분의 이전 값과 이후 값을 확인
    for i in range(len(continuous)):
        idx1 = continuous[i][0]-1
        idx2 = continuous[i][1]+2
        # print(i+1, '번------------------------------\n', data['논산배수지'].iloc[idx1:idx2])

        # 연속되지 않은 부분은 이전 값으로 대체 
    for i in discontinuous:
        data['논산배수지'].iloc[i] = data['논산배수지'].iloc[(i-1)]
        # print(f'{i}행', data['논산배수지'].iloc[i])
    
    difference = pd.DataFrame({'유출유량 변화':data['논산배수지'] - data['논산배수지'].shift(1)})
    # print(len(get_outlier(difference, '유출유량 변화')))
    for i in get_outlier(difference, '유출유량 변화'):
        if difference.iloc[i, 0] < 0:  # 변화값이 음수일 경우
            data['논산배수지'].iloc[(i-1)] = data['논산배수지'].iloc[(i-2)]        
        else:  # 변화값이 양수일 경우
            data['논산배수지'].iloc[i] = data['논산배수지'].iloc[(i-1)]
    request_json_data=json.dumps(dict(data['논산배수지'])) #add
    with open("output.json", "w") as json_file:
        json_file.write(request_json_data)
    target_col = data.filter(['논산배수지'])
    target_col_values = target_col.values
    print(target_col_values)
    # 데이터의 80%의 인덱스 수
    train_len = math.ceil(len(target_col_values) * 0.8)  # 29810
    # 8:2로 데이터 분할
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
    # array로 변환

    # array(): array 데이터를 생성하는 함수
    train_x_array = np.array(x_train)
    train_y_array = np.array(y_train)
    test_x_array = np.array(x_test)

    # lstm학습하기 위해 reshape

    # reshape(): array의 차원을 변형
    train_x_array = np.reshape(train_x_array, (train_x_array.shape[0], train_x_array.shape[1], 1))  
    test_x_array = np.reshape(test_x_array, (test_x_array.shape[0], test_x_array.shape[1], 1)) 
    # print(train_x_array.shape)
    # print(train_y_array.shape)
    # print(test_x_array.shape)
    # print(y_test.shape)
    # print(train_y_array)
    return {"train_x": train_x_array, "train_y": train_y_array, "test_x": test_x_array, "test_y": y_test, "train_df": train_df, "test_df": test_df, "data": data}
    

def train():
    logger.info("Preparing dataset...")
    dataset = prepare_dataset()
    y_train=dataset["train_y"]
    x_train=dataset["train_x"]
    y_test=dataset["test_y"]
    x_test=dataset["test_x"]
    train_df=dataset["train_df"]
    test_df=dataset["test_df"]
    data=dataset["data"]
    train_min = np.min(train_df)
    train_max = np.max(train_df)

    logger.info("Training model...")
    model_lstm = Sequential()
    model_lstm.add(LSTM(100, return_sequences = False, input_shape = (x_train.shape[1], 1)))  # input_shape = (100,1)  
    model_lstm.add(Dense(25, activation = 'relu'))  # Dense(): 출력 뉴런 수, activation = 'relu': 활성화함수 설정
    model_lstm.add(Dense(1))
    
    model_lstm.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy']) 
    history = model_lstm.fit(x_train, y_train, batch_size = 16, epochs = 1)
    pred_lstm_result = model_lstm.predict(x_train) 
    pred_lstm_result_restored = np.array([])  # predictions
    for i in range(len(train_df)-168):
        pred_lstm_result_restored = np.append(pred_lstm_result_restored, ((pred_lstm_result[i]*(train_max - train_min)) + train_min), axis = 0)
    print(x_train.shape) # (29642, 168, 1)
    # print(x_test.shape)
    print(pred_lstm_result_restored.shape)  # (29642, )
    # return 1
    train_y_restored = np.array([])  # y_train
    for i in range(len(train_df)-168):
        train_y_restored = np.append(train_y_restored, ((y_train.reshape(29642,1)[i]*(train_max - train_min)) + train_min))
    # 평가지표 계산과 df만들기 위해 reshape
    pred_lstm_result_restored_reshape = pred_lstm_result_restored.reshape(29642)
    train_y_reshape = train_y_restored.reshape(29642)
    # 결과 df만들기

    # DataFrame(데이터): DataFrame생성하는 함수
    # transpose(): 행열 전환(전치)시키는 함수
    # columns: 컬럼명 출력
    train_prediction_df = pd.DataFrame(data = [train_y_reshape, pred_lstm_result_restored_reshape])  # df생성
    train_prediction_df = train_prediction_df.transpose()  # 행열전환
    train_prediction_df['연월일시'] = data['연월일시'][:29642]  # '연월일시'열을 추가 
    train_prediction_df.columns = ['train_y_reshape', 'pred_lstm_result_restored', '연월일시']  # 컬럼명 변경
    train_prediction_df = train_prediction_df[['연월일시', 'train_y_reshape', 'pred_lstm_result_restored']]  # 컬럼순서 변경
    # train_prediction_df.to_csv('output/train_prediction.csv', encoding='utf-8-sig', index = None)  # 실제값, 예측값 저장
    train_prediction_df.head(2)  # 위에서부터 2행출력
    # 모델 평가 지표 생성 및 평가

    # mean absolute error (MAE)
    # str(): 데이터를 문자열로 바꿔주는 함수
    # round(): 숫자 데이터를 반올림

    # lstm 모형의 평가지표
    train_mae = mean_absolute_error(pred_lstm_result_restored_reshape, train_y_reshape)
    train_rmse = np.sqrt(mean_squared_error(pred_lstm_result_restored_reshape, train_y_reshape))
    print('LSTM MAE: ' + str(round(train_mae, 1)))
    print('LSTM RMSE: ' + str(round(train_rmse, 1)))

    # 일주일 전의 값으로 현재의 유출유량을 예측한 값의 평가지표
    week_mae = ((abs(data['논산배수지']-data['논산배수지'].shift(168)).sum())/len(data))  # 24*7=168
    week_rmse = np.sqrt((((data['논산배수지']-data['논산배수지'].shift(168))**2)/len(data)).sum())
    print('1WEEK MAE: ' + str(round(week_mae, 1)))
    print('1WEEK RMSE: '+ str(round(week_rmse, 1)))

    # 모델 예측
    pred_lstm_test_result = model_lstm.predict(x_test)
    # scaling 복구
    pred_lstm_test_result_restored = np.array([])
    for i in range(len(test_df)-168):
        pred_lstm_test_result_restored = np.append(pred_lstm_test_result_restored,(pred_lstm_test_result[i]*(train_max-train_min)) + train_min, axis = 0)

    y_test_restored = np.array([])  # y_train
    for i in range(len(test_df)-168):
        y_test_restored = np.append(y_test_restored, ((y_test.reshape(7452,1)[i]*(train_max-train_min)) + train_min))
    # 평가지표 계산과 df만들기 위해 reshape
    pred_lstm_test_result_restored_reshape = pred_lstm_test_result_restored.reshape(7452)
    y_test_reshape = y_test_restored.reshape(7452)

    # lstm 모형의 평가지표
    mae_test = mean_absolute_error(pred_lstm_test_result_restored_reshape, y_test_reshape)
    rmse_test = np.sqrt(mean_squared_error(pred_lstm_test_result_restored_reshape, y_test_reshape))
    print('LSTM MAE: ' + str(round(mae_test, 1)))
    print('LSTM RMSE: ' + str(round(rmse_test, 1)))

    # 일주일 전의 값으로 현재의 유출유량을 예측한 값의 평가지표
    week_mae = ((abs(data['논산배수지']-data['논산배수지'].shift(168)).sum())/len(data))  # 24*7=168
    week_rmse = np.sqrt((((data['논산배수지']-data['논산배수지'].shift(168))**2)/len(data)).sum())
    print('1WEEK MAE: ' + str(round(week_mae, 1)))
    print('1WEEK RMSE: '+ str(round(week_rmse, 1)))
    logger.info("Saving artifacts...")
    Path("artifacts").mkdir(exist_ok=True)
    dump(model_lstm, "artifacts/model.joblib")
    # dump(scaler, "artifacts/scaler.joblib")
    # logger.info(f"Test MSE: {error}")


if __name__ == "__main__":
    # prepare_dataset()
    train()

