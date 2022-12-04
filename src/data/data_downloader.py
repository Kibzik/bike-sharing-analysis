import os
import zipfile
import pandas as pd

from wwo_hist import retrieve_hist_data


class DataDownloader:


    def get_trips_data(self):
        df_trips = pd.DataFrame()
        files_to_remove = []
        url = "https://divvy-tripdata.s3.amazonaws.com/"
        download_folder = "../../data/trips/"
        years = ['2020', '2021', '2022']
        monthes = ['0' + str(m) if m < 10 else str(m) for m in range(0, 13, 1)]
        for year in years:
            for month in monthes:
                code = os.system(f"wget -P {download_folder} {url}{year}{month}-divvy-tripdata.zip")
                if not code:
                    zf = zipfile.ZipFile(f"{year}{month}-divvy-tripdata.zip")
                    trip_df = pd.read_csv(zf.open(zf.infolist()[0].filename))
                    df_trips.append(trip_df)
                    files_to_remove.append(f"{year}{month}-divvy-tripdata.zip")
        [os.remove(ftr) for ftr in files_to_remove]

    def get_weather_data(self):
        api_key = '54951602af4b445ca3f180347222811'
        location_list = ['chicago']
        start_date = '01-APR-2020'
        end_date = '27-NOV-2022'
        frequency = 1  # hour diff

        hist_weather_data = retrieve_hist_data(api_key,
                                               location_list,
                                               start_date,
                                               end_date,
                                               frequency,
                                               location_label=False,
                                               export_csv=True,
                                               store_df=True)