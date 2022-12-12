import yaml


def read_config(config_path: str) -> dict:
    """
    Reads config from .yaml file.
    :param config_path: path to .yaml file to load from

    :return: configuration dictionary
    """
    with open(config_path, "rb") as f_in:
        config = yaml.safe_load(f_in)
    return config


def parse_data_config(config: dict) -> dict:
    """
    Parses the data configuration dictionary.
    :param config: data configuration dictionary

    :return: data configuration parameters
    """
    params = dict()
    params["trips_url"] = config["trips_url"]
    params["trips_folder"] = config["trips_folder"]
    params["wwo_hist_folder"] = config["wwo_hist_folder"]
    params["api_key"] = config["api_key"]
    params["location_list"] = config["location_list"]
    params["start_date"] = config["start_date"]
    params["end_date"] = config["end_date"]
    params["frequency"] = config["frequency"]

    return params


def parse_train_config(config: dict) -> dict:
    """
    Parses the model configuration dictionary.
    :param config: model configuration dictionary

    :return: model configuration parameters
    """
    params = dict()
    params["input_data_path"] = config["input_data_path"]
    params["output_encoder_path"] = config["output_encoder_path"]
    params["output_scaler_path"] = config["output_scaler_path"]
    params["output_model_path"] = config["output_model_path"]
    params["metric_path"] = config["metric_path"]
    params["split_test_size"] = config["splitting_params"]["test_size"]
    params["split_random_state"] = config["splitting_params"]["random_state"]

    params["model_type"] = config["train_params"]["model_type"]
    params["model_random_state"] = config["train_params"]["random_state"]
    params["n_estimators"] = config["train_params"]["n_estimators"]
    params["min_child_weight"] = config["train_params"]["min_child_weight"]
    params["max_depth"] = config["train_params"]["max_depth"]
    params["learning_rate"] = config["train_params"]["learning_rate"]

    return params
