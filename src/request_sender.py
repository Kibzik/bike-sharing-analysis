import requests
import pandas as pd

from time import sleep


def send_requests(host: str = "localhost:5050", dataset_path=None) -> None:
    """
    Generates requests from the file and sends it to the host.
    :param host: hostname:port or URL address
    :param dataset_path: dataset_path to data to generate request

    :return: none

    """
    data = pd.read_csv(dataset_path, index_col=[0])
    for request_data in data.to_dict("records"):
        print(f"Request to service: {request_data}")
        response = requests.post(url=f"http://{host}/predict", json=request_data)
        print(f"Status code: {response.status_code}")
        print(f"Result: {response.text}")
        sleep(1)


if __name__ == "__main__":
    dataset_path = "../data/request.csv"
    send_requests(dataset_path=dataset_path)
