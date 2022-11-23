import hydra
from omegaconf import DictConfig

import src.runner as runner
import src.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="dci")
def dci_metric(cfg: DictConfig):
    utils.display_config(cfg)
    runner.dci_metric(cfg)


if __name__ == "__main__":
    dci_metric()
