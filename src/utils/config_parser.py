import yaml


def parse_config(config_path):
    """
    Parse a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
