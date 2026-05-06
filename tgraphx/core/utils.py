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


def get_device(device_id: int = 0) -> torch.device:
    r"""Return the best available device.

    Priority: CUDA > Apple Silicon MPS > CPU.

    Args:
        device_id (int): CUDA device index. Ignored for MPS and CPU. Default: 0.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{device_id}')
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')
    return torch.device('cpu')
