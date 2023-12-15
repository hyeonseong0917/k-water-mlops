from locust import HttpUser, task
import pandas as pd
import requests

dataset = (
    pd.read_csv('kwater_recipe07_dataset.csv')
)
class ReservoirPrediction(HttpUser):
    @task(1)
    def healthcheck(self):
        self.client.get("/healthcheck")

    @task(10)
    def prediction(self):
        # record = random.choice(dataset).copy()
        response=requests.get("http://3.236.249.63:5000/transform")
        json_response = response.json()
        json_data=json_response["data"]
        new_json_data={"data": json_data}
        self.client.post("/predict", json=new_json_data)

    # @task(2)
    # def prediction_bad_value(self):
    #     record = random.choice(dataset).copy()
    #     corrupt_key = random.choice(list(record.keys()))
    #     record[corrupt_key] = "bad data"
    #     self.client.post("/predict", json=record)