import logging
import sys
import os
import pandas as pd
import holidays
import numpy as np

from math import radians, cos, sin, asin, sqrt

from src.config_reader import read_config, parse_data_config
from src.data.data_downloader import DataDownloader


class DatasetCreator:
    def __init__(self, data_params: dict = None):
        self.data_params = data_params
        self.holidays: list = [holiday[0] for holiday in holidays.USA(years=range(int(
            self.data_params["start_date"][-4:]), int(self.data_params["end_date"][-4:]) + 1, 1)).items()]
        self.month2season: dict = {1: 1,
                                   2: 1,
                                   3: 2,
                                   4: 2,
                                   5: 2,
                                   6: 3,
                                   7: 3,
                                   8: 3,
                                   9: 4,
                                   10: 4,
                                   11: 4,
                                   12: 1}

    @staticmethod
    def get_data(directory: str):
        df_ = pd.DataFrame()
        data_filenames = [filename for filename in os.listdir(
            directory) if filename.endswith(".csv")]
        for filename in data_filenames:
            temp_df = pd.read_csv(directory + filename)
            df_ = pd.concat([df_, temp_df])
        return df_

    def create_time_features(self, df_, date_col="date_time"):
        df_[date_col] = pd.to_datetime(df_[date_col])

        df_["year"] = df_[date_col].dt.year
        df_["month"] = df_[date_col].dt.month
        df_["season"] = df_["month"].apply(lambda x: self.month2season[x])
        df_["day"] = df_[date_col].dt.day
        df_["hour"] = df_[date_col].dt.hour
        df_["week_day"] = df_[date_col].dt.day_name()
        df_["is_weekend"] = np.where(
            (df_["week_day"] == "Saturday") | (df_["week_day"] == "Sunday"), 1, 0)

        df_["is_holiday"] = np.where(
            df_[date_col].dt.date.isin(self.holidays), 1, 0)
        return df_

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        r = 6371
        return c * r

    @staticmethod
    def remove_outliers(df_):
        # calculate the percentiles
        ride_length_q1 = df_["ride_length"].quantile(0.25)
        ride_length_q3 = df_["ride_length"].quantile(0.75)
        ride_length_iqr = ride_length_q3 - ride_length_q1

        trip_duration_q1 = df_["trip_duration"].quantile(0.25)
        trip_duration_q3 = df_["trip_duration"].quantile(0.75)
        trip_duration_iqr = trip_duration_q3 - trip_duration_q1

        return df_[(df_["ride_length"] > ride_length_q1 - 1.5 * ride_length_iqr) &
                   (df_["ride_length"] < ride_length_q3 + 1.5 * ride_length_iqr) &
                   (df_["trip_duration"] > trip_duration_q1 - 1.5 * trip_duration_iqr) &
                   (df_["trip_duration"] < trip_duration_q3 + 1.5 * trip_duration_iqr)]


def dataset_creating(data_params: dict = None, logger: logging = None) -> pd.DataFrame:
    # download data
    # DataDownloader(data_params=data_params)

    dc = DatasetCreator(data_params)
    # read data
    logger.info("Read trips data...")
    df_trips = dc.get_data(data_params["trips_folder"])
    print(
        f"    df_trips has {df_trips.shape[0]} observations and {df_trips.shape[1]} features")
    logger.info("Read weather data...")
    df_weather = dc.get_data(data_params["wwo_hist_folder"])
    print(
        f"    df_weather has {df_weather.shape[0]} observations and {df_weather.shape[1]} features")
    df_weather = df_weather[["date_time", "maxtempC", "mintempC", "totalSnow_cm",
                             "FeelsLikeC", "cloudcover", "humidity", "pressure", "visibility", "windspeedKmph"]]

    logger.info("Validation of the initial dataset...")
    # drop miss values
    logger.info("   Drop missing values...")
    if df_trips.isna().sum().sum():
        df_trips.dropna(inplace=True)
    logger.info("   Done!")
    # create new columns
    logger.info("   Create new columns...")
    df_trips = dc.create_time_features(df_trips, "started_at")
    df_trips["ended_at"] = pd.to_datetime(df_trips["ended_at"])
    df_trips["trip_duration"] = df_trips.apply(lambda x: (
        x["ended_at"] - x["started_at"]).total_seconds(), axis=1)
    df_trips["ride_length"] = df_trips.apply(lambda x: dc.haversine(
        x["start_lat"], x["start_lng"], x["end_lat"], x["end_lng"]), axis=1)
    df_trips.drop(["started_at", "ended_at"], inplace=True, axis=1)
    df_weather = dc.create_time_features(df_weather, "date_time")
    logger.info("   Done!")
    # merge datasets
    logger.info("   Merge datasets...")
    df_trips = df_trips.merge(df_weather, on=[
        "year", "month", "season", "day", "hour", "week_day", "is_weekend", "is_holiday"])
    logger.info("   Done!")
    # remove outliers
    logger.info("   Remove outliers...")
    df_trips = dc.remove_outliers(df_trips)
    df_trips.reset_index(drop=True, inplace=True)
    logger.info("   Done!")
    logger.info("Dataset was successfully validated.")

    # group dataset for final df
    logger.info("Make final df...")
    df_trips = df_trips.groupby(
        ["FeelsLikeC", "maxtempC", "mintempC", "windspeedKmph", "cloudcover", "humidity", "pressure",
         "visibility", "is_holiday", "is_weekend", "year", "season", "month", "hour", "day", "week_day"]) \
        .agg({"start_station_name": "size"}) \
        .rename(columns={"start_station_name": "number_of_rides"}) \
        .reset_index()
    print(
        f"    df has {df_trips.shape[0]} observations and {df_trips.shape[1]} features")

    return df_trips


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    config = read_config("../../configs/data_config.yaml")
    data_params = parse_data_config(config)
    data_params["trips_folder"] = "../../data/trips/"
    data_params["wwo_hist_folder"] = "../../data/weather/"

    df_trips = dataset_creating(data_params, logger)
    df_trips.to_csv("../../data/df_for_modelling.csv")
