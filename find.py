import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="find")
def find_directions(cfg: DictConfig):
    utils.display_config(cfg)
    runner.find_directions(cfg)


if __name__ == "__main__":
    find_directions()
