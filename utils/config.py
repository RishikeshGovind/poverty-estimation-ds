import yaml
from pathlib import Path

_config = None

def load_config(path: str = None) -> dict:
    global _config
    if _config is not None:
        return _config
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config
