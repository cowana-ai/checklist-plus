import os

from hydra import compose, initialize
from omegaconf import OmegaConf


def load_config(verbose=False):
    with initialize(version_base="1.3.2", config_path="."):
        cfg = compose(config_name="config")
        if verbose:
            print(OmegaConf.to_yaml(cfg))
        return cfg

cfg = load_config()
