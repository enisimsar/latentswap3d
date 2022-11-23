import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="tune")
def tune_parameters(cfg: DictConfig):
    utils.display_config(cfg)
    runner.tune_parameters(cfg)


if __name__ == "__main__":
    tune_parameters()
