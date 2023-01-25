import logging
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.exptool import (
    prepare_trainer_config,
    print_config,
    register_omegaconf_resolver,
    try_resume,
)

register_omegaconf_resolver()
logger = logging.getLogger(__name__)

pl._logger.handlers = []
pl._logger.propagate = True

dotenv.load_dotenv(override=True)


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:

    # Assign hydra.run.dir=<previous_log_dir> to resume training.
    # If previous checkpoints are detected, `cfg` will be replaced with the
    # previous config
    cfg = try_resume(cfg)

    # Print & save config to logdir
    print_config(cfg, save_path="config.yaml")

    # Set random seed
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    # Initialize datamodule
    datamodule = instantiate(cfg.dataset)

    # Initialize pipeline
    pipeline = instantiate(cfg.pipeline, cfg=cfg, _recursive_=False)

    # Initialize trainer
    cfg_trainer = prepare_trainer_config(cfg)
    trainer = pl.Trainer(**cfg_trainer)

    # Training
    if cfg.resume_ckpt is not None:
        logger.info(f"resume from {cfg.resume_ckpt}")
    trainer.fit(pipeline, datamodule, ckpt_path=cfg.resume_ckpt)

    # Testing
    if cfg.run_test:
        trainer.test(pipeline, datamodule, ckpt_path=None)

    # print logdir for conveniently copy
    logger.info(f"Logdir: {Path('.').resolve()}")


if __name__ == "__main__":
    main()
