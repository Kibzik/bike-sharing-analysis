# Bike Sharing Prediciton
![bike-sharing](https://raw.githubusercontent.com/Masterx-AI/Project_BoomBikes_Share_Prediction/main/bbk.jpg)
## Table of Contents
 * [Problem Description](#problem-description)
   * [About the Data](#about-the-data)
   * [Questions](#questions)
   * [Goals](#goals)
   * [Features Description](#features-description)
 * [Tech Stack and concepts](#tech-stack-and-concepts)
 * [Setup](#setup)
   * [Virtual Environment](#virtual-environment)
   * [Run Service Locally](#run-service-locally)
   * [Docker Container](#docker-container)
   * [Deploying to Cloud](#deploying-to-cloudheroku-docker-deployment)

## Problem Description
The problem we will study was introduced by [Divvy Bikes](https://divvybikes.com/). the bicycle sharing system in the Chicago metropolitan area, currently serving the cities of Chicago and Evanston.
I. Company believes the future success depends on maximizing the number of annual memberships. Understand how casual riders and annual members use bikes differently. Design a new marketing strategy to convert casual riders into annual members.  

II. The director of logistic gave a task to find out the places where they need to stop let bikes and the places where to reinforce the number on bikes. Investigate the GIS aspect of members and casuals. Find  "hot" points of city where we need to bring more bicycles and where to take.  

III. You've been told this year has been a "disaster": almost all the bicycles are out of commission, some are under repair. We need to know the approximate number of bicycles for the next month.

### Questions
**I.**
- How do annual members and casual riders use bikes differently?
- Why would casual riders buy annual memberships?
- How can company use digital media to influence casual riders to become members?
---
**II.**
- What are the "hot"/"cold" spots of the city?
- Are the majority of routes intercommunity or intracommunity?
----
**III.**
- What is the count of bicycles we need to prepare based on data?

### Goals
- Determine the factors that influence casual riders into buying annual memberships

- Identify historical trends for casual and annual bike riders

- Use insights from historical trends and factors associated with casual riders buying annual memberships to improve the casual rider to annual membership conversion rate via digital media.

- Find "hot"/"cold" spots in the city in order to deliver or take out bicycles from these spots.

- Develop a model to find approximate number of bicycles for every day.

### About the Data
1. Company makes its Historical trip data available for public use. The datasets were downloaded from [link](https://divvy-tripdata.s3.amazonaws.com/index.html), under this [license](https://ride.divvybikes.com/data-license-agreement).
Each trip is anonymized and includes, trip start day and time, trip end day and time, Trip start station, a Trip end station, Rider type. 
For this project, I will be analyzing Cyclist trip data between *January, 2021 and November, 2022*. Each month's data in a separate CSV file was loaded and were later concatinated.
2. Weather data was taken using [world weather API](https://www.worldweatheronline.com/weather-api/api/)
3. Geo data of Chicago was taken from [shape files for the boundaries of the city of Chicago](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-City/ewy2-6yfk)

### Features Description
* **AGE**: Age of the employee
* **FeelsLikeC**: what the temperature feels like
* **maxtempC**: maximum temperature of the day
* **mintempC**: minimum temperature of the day
* **windspeedKmph**: wind speed in Km per hour
* **cloudcover**: level of cloudcover
* **humidity**: level of humidity
* **pressure**: level of pressure
* **visibility**: level of visibility
* **is_holiday**: (1 - holiday, 0 - not)
* **is_weekend**: (1 - weekend, 0 - not)
* **year**: 
* **season**: season of the year
* **month**: month of the year
* **hour**: hour of the day
* **day**: day of the month
* **week_day**: day name of the week

## Tech Stack and concepts

- Python
- Scikit-learn
- XGBoost
- Machine Learning Pipeline
- FastAPI
- Virtual environment
- Docker
- Heroku

## Setup
Clone the project repo and open it.

If you want to reproduce results by running [notebooks](notebooks/) or [`train.py`](src/train.py), 
you need to download data, create a virtual environment and install the dependencies.

### Download data
For notebooks:
1. To download data use [this notebook](notebooks/download_data.ipynb) or [this script](src/data/data_downloader.py)
2. When you run notebook be ready that it'll eat memory and take 10-15 minutes of your time
3. For answers based on GIS in you need to download shapefiles from [shape files for the boundaries of the city of Chicago](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-City/ewy2-6yfk) - Export -> shapefile,
 unzip them and move to data/geo

### Virtual Environment
In case of `conda`(you feel free to choose any other tools (`pipenv`, `venv`, etc.)), just follow the steps below:
1. Open the terminal and choose the project directory.
2. Create new virtual environment by command `conda create -n test-env python=3.10`.
3. Activate this virtual environment with `conda activate test-env`.
4. Install all packages using `pip install -r requirements.txt`.

### Run service locally
To run the service locally in your environment, simply use the following commands:
- Windows
```bash
waitress-serve --listen=0.0.0.0:5050 predict:app
```
- Ubuntu
```bash
gunicorn predict:app --host=0.0.0.0 --port=5050
```

### Containerization
Be sure that you have already installed the Docker, and it's running on your machine now.
1. Open the terminal and choose the project directory.
2. Build docker image from [`Dockerfile`](Dockerfile) using `docker build --no-cache -t predict-cnt-riders .`.
With `-t` parameter we're specifying the tag name of our docker image. 
3. Now use `docker run -it -p 5050:5050 predict-cnt-riders` command to launch the docker container with your app. 
Parameter `-p` is used to map the docker container port to our real machine port.

Also you can pull out an already built image from [Dockerhub](https://hub.docker.com/). 
1. Use this command `docker pull kibzikm/predict-cnt-riders:latest` in this case.
2. Now use `docker run -it -p 5050:5050 kibzikm/predict-cnt-riders` command to launch the docker container with your app.

### Deploying to Cloud(Heroku docker deployment)
Follow this steps to deploy the app to Heroku
1. Register on [Heroku](https://signup.heroku.com/) and install Heroku CLI.
2. Open the terminal in project of the app
3. Terminal: rung the `heroku login` command to log in to Heroku.
4. Terminal: login to Heroku container registry using `heroku container:login` command.
5. Terminal: create a new app in Heroku with the following command `heroku create predict-cnt-riders-docker`.
6. Make small changes in [`Dockerfile`](Dockerfile): uncomment the last line and comment out the line above. 
Heroku automatically assigns porn number from the dynamic pool. So, there is no need to specify it manually.
7. Terminal: run the `heroku container:push web -a predict-cnt-riders-docker` command to push docker image to Heroku.
8. Terminal: release the container using the command `heroku container:release web -a predict-cnt-riders-docker`.
9. Launch your app by clicking on generated URL in 5th step. In our case the link - [Heroku app](https://predict-cnt-riders-docker.herokuapp.com/).
If we have successfully deployed the app, the link opens without problems.

Now we can move on to the next step - service testing.
### Service testing
* [prediction endpoint](https://parking-slots-docker.herokuapp.com/predict) serves for the model scoring.

To test the prediction endpoint you can use handmade script  [request sender](src/request_sender.py) that takes data from a specified directory and sends requests to the service.
In case of using a script, just follow the rule:

To test the service that is running locally
- Just run the  [request sender](src/request_sender.py) without any changes

To test our Heroku deployment, we should type:
- Set host parameter in 27 line to 'predict-cnt-riders.herokuapp.com' and run [request sender](src/request_sender.py)