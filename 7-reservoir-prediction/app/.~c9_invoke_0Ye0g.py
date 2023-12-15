import os
from typing import Callable

import numpy as np
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info
import json
import requests
from sklearn.metrics import mean_squared_error

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)
## ENABLE_METRICS 가 true 인 Runtime 에서만 동작하게 됨 

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
def regression_model_output(
    metric_name: str = "regression_model_output",
    metric_doc: str = "Output value of regression model",
    metric_namespace: str = "",
    metric_subsystem: str = "",
     buckets=(*np.arange(0, 500, 10).tolist(), float("inf")),
) -> Callable[[Info], None]:
    METRIC = Histogram(
        metric_name,
        metric_doc,
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            # predicted_quality = info.response.headers.get("X-model-score")
            # if predicted_quality:
            #     METRIC.observe(float(predicted_quality))
            status_code = info.response.status_code
            print(info.response.body)
            content = None
            
            try:
                content = json.loads(info.response.body.decode("utf-8"))
            except Exception as e:
                # print(e)
                content=None    
                
            
            # 이 정보를 사용하여 원하는 로직을 수행할 수 있습니다.
            print("why")
            print(content)
            if status_code == 200 and content:
                # content를 가공하거나 원하는 작업을 수행합니다.
                # 예를 들어, content를 JSON으로 파싱하여 값을 얻거나 다른 처리를 수행할 수 있습니다.
                print("hello")
                try:
                    json_data = json.loads(content)
                    predicted_quality = json_data.get("data")
                    true_values=get_true_values()
                    print("hi")
                    if predicted_quality and true_values:
                        rmse_value = np.sqrt(mean_squared_error(true_values, predicted_quality))
                        print(rmse_value)
                        METRIC.observe(rmse_value)
                        # METRIC.observe(float(predicted_quality))
                except json.JSONDecodeError as e:
                    # JSON 디코딩 오류 처리
                    print(f"Error decoding JSON: {e}")

    return instrumentation

def get_true_values():
    response=requests.get("http://34.239.179.173:5000/test")
    content=response.json()
    return content["data"]
    
buckets = (*np.arange(0, 500, 10).tolist(), float("inf"))
# instrumentator.add(
#     regression_model_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM, buckets=buckets)
# )
# buckets = np.arange(0, 1.1, 0.01).tolist() + [float("inf")]
instrumentator.add(
    regression_model_output(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM, buckets=buckets)
)
