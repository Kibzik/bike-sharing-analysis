import logging
import sys
import pickle

import uvicorn
from fastapi import FastAPI

from src.bike_sharing_description import BikeSharingData, BikeSharingResponse


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_object(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


encoder = load_object('models/encoder.pkl')
scaler = load_object('models/scaler.pkl')
model = load_object('models/model.pkl')

app = FastAPI(title='Number of rides prediction')


@app.get('/')
def main():
    return 'Welcome to the number of rides prediction webservice!'


@app.post('/predict', response_model=BikeSharingResponse)
def predict(request: BikeSharingData):
    print(f'Incoming request:\n{request}')
    bike_sharing_features = request.dict()
    X = encoder.transform(bike_sharing_features)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return BikeSharingResponse(prediction=y_pred)


if __name__ == "__main__":
    uvicorn.run('predict:app', host='0.0.0.0', port=5050, reload=True)
