import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="gen")
def gen(cfg: DictConfig):
    utils.display_config(cfg)
    runner.generate(cfg)


if __name__ == "__main__":
    gen()
