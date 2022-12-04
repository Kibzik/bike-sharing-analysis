FROM python:3.11.0

WORKDIR /user/app
COPY requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY data/request.csv ./data/request.csv
COPY models/ ./models/
COPY src/bike_sharing_description.py ./src/bike_sharing_description.py
COPY predict.py ./

EXPOSE 5050
CMD uvicorn predict:app --host=0.0.0.0 --port=5050
