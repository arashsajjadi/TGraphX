import json
import torch

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path):
    r"""Load configuration from a YAML or JSON file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Configuration parameters.
    """
    if config_path.endswith(('.yaml', '.yml')):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configuration files.")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use YAML or JSON.")
    return config


def get_device():
    r"""Return the available device ('cuda' if available, else 'cpu')."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
