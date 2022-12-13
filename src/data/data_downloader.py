import os
import zipfile
import pandas as pd

from src.config_reader import read_config, parse_data_config

from wwo_hist import retrieve_hist_data


class DataDownloader:
    def __init__(self, data_params: dict = None):
        self.data_params = data_params
        self.get_trips_data()
        self.get_weather_data()

    def get_trips_data(self):
        files_to_remove = []
        download_folder = self.data_params["trips_folder"]
        years = range(int(self.data_params["start_date"][-4:]),
                      int(self.data_params["end_date"][-4:]) + 1, 1)
        months = ["0" + str(m) if m < 10 else str(m) for m in range(0, 13, 1)]
        for year in years:
            for month in months:
                code = os.system(
                    f"wget -P {download_folder} {self.data_params['trips_url']}{year}{month}-divvy-tripdata.zip")
                if not code:
                    zf = zipfile.ZipFile(
                        f"{download_folder}/{year}{month}-divvy-tripdata.zip")
                    trip_df = pd.read_csv(zf.open(zf.infolist()[0].filename))
                    trip_df.to_csv(
                        f"{download_folder}/{year}{month}-divvy-tripdata.csv", index=False)
                    files_to_remove.append(
                        f"{download_folder}/{year}{month}-divvy-tripdata.zip")
        try:
            [os.remove(ftr) for ftr in files_to_remove]
        except PermissionError as pe:
            print("Error in remove file: " + str(pe))

    def get_weather_data(self):
        os.chdir(self.data_params["wwo_hist_folder"])
        hist_weather_data = retrieve_hist_data(self.data_params["api_key"],
                                               self.data_params["location_list"],
                                               self.data_params["start_date"],
                                               self.data_params["end_date"],
                                               self.data_params["frequency"],
                                               location_label=False,
                                               export_csv=True,
                                               store_df=True)


if __name__ == "__main__":
    config = read_config("../../configs/data_config.yaml")
    data_params = parse_data_config(config)
    data_downloader = DataDownloader(data_params=data_params)
