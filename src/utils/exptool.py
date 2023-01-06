import logging
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from rich.panel import Panel
from rich.syntax import Syntax

logger = logging.getLogger(__name__)


def register_omegaconf_resolver():
    OmegaConf.register_new_resolver(
        "tail", lambda x: x.split(".")[-1], replace=True
    )


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "dataset",
        "pipeline",
        "model",
        "criterion",
        "optim",
        "scheduler",
        "pl_trainer",
        "callbacks",
        "logging",
        "logdir",
    ),
    resolve: bool = True,
    save_path: str = None,
) -> None:

    ordered_config = OrderedDict()
    for key in print_order:
        if key in config:
            ordered_config[key] = config[key]
    for key in config:
        if key not in ordered_config:
            ordered_config[key] = config[key]
    ordered_config = OmegaConf.create(dict(ordered_config))
    content = OmegaConf.to_yaml(ordered_config, resolve=resolve)

    panel = Panel(
        Syntax(
            content,
            "yaml",
            background_color="default",
            line_numbers=True,
            code_width=80,
        ),
        title="Config",
        expand=False,
    )
    rich.print()
    rich.print(panel)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(content)


class Experiment:
    def __init__(self, logdir, wandb_entity=None) -> None:
        self.logdir = Path(logdir).expanduser()
        self._wandb_entity = wandb_entity
        self._config = None

    @property
    def wandb_entity(self):
        if self._wandb_entity is None:
            import wandb

            self._wandb_entity = wandb.Api().default_entity
            logger.info(
                "wandb_entity is None, use default entity: {}".format(
                    self._wandb_entity
                )
            )
        return self._wandb_entity

    @property
    def config(self):
        if self._config is None:
            self._config = self._load_config()
        return self._config

    @property
    def config_path(self):
        return self.logdir / "config.yaml"

    @property
    def hydra_raw_config_path(self):
        return self.logdir / ".hydra" / "config.yaml"

    @property
    def hydra_config_path(self):
        return self.logdir / ".hydra" / "hydra.yaml"

    @property
    def wandb_run_path(self):
        return f"{self.wandb_entity}/{self.project}/{self.wandb_run_id}"

    @property
    def wandb_run_id(self):
        return self.config.logging.wandb.version

    @property
    def project(self):
        return self.config.project

    @property
    def ckpt_dir(self):
        return self.logdir / "checkpoints"

    @property
    def best_ckpt_path(self):
        ckpts = [
            ckpt
            for ckpt in list(sorted(self.ckpt_dir.glob("*.ckpt")))
            if "last" not in ckpt.name
        ]
        return ckpts[-1]

    @property
    def last_ckpt_path(self):
        return self.ckpt_dir / "last.ckpt"

    def ckpt_path(self, ckpt_name):
        if self.ckpt_dir / f"{ckpt_name}.ckpt".exists():
            return self.ckpt_dir / f"{ckpt_name}.ckpt"
        elif self.ckpt_dir / f"{ckpt_name}".exists():
            return self.ckpt_dir / f"{ckpt_name}"
        else:
            return None

    def _load_config(self):
        config = None
        if self.config_path.exists():
            config = OmegaConf.load(self.config_path)
        elif self.hydra_raw_config_path.exists():
            # try to resolve time from the logdir
            time_str = str(self.logdir).split(".")[-1]
            time_fmt = "%Y-%m-%d_%H-%M-%S"
            log_time = datetime.strptime(time_str, time_fmt)

            # resolve hydra config from raw config
            OmegaConf.register_new_resolver("now", log_time.strftime)
            config = OmegaConf.load(self.hydra_raw_config_path)
            logger.info(
                f"config has been read from {self.hydra_raw_config_path}"
            )
        return config

    def get_pipeline_model_loaded(self, ckpt="last", config=None):
        if config is None:
            config = self.config
        if ckpt == "best":
            ckpt_path = self.best_ckpt_path
        elif ckpt == "last":
            ckpt_path = self.last_ckpt_path
        elif self.ckpt_path(ckpt) is not None:
            ckpt_path = self.ckpt_path(ckpt)
        else:
            raise ValueError(f"ckpt {ckpt} does not exist")

        logger.info(f"loading {ckpt_path}")
        pipeline_cfg = dict(config["pipeline"])
        pipeline_cfg[
            "_target_"
        ] = f"{pipeline_cfg['_target_']}.load_from_checkpoint"
        pipeline_cfg["checkpoint_path"] = ckpt_path
        # set LightningModule.cfg from hparams_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(OmegaConf.to_yaml(config, resolve=True))
            f.flush()
            pipeline_cfg["hparams_file"] = f.name
            model = instantiate(pipeline_cfg)
        return model


def element_exists(element):
    try:
        _ = element
        return True
    except Exception:
        return False


def try_resume(cfg, replace_wandb_only=False):
    """Resume from previous checkpoint.

    Note
    ----------
    Usage: run commands like:
    ```
    python train.py hydra.run.dir=<previous logdir>
    ```

    Triggered when finding previous last_ckpt_path. It will replace the config
    with previous config and load last checkpoint.
    Wandb run will be also resumed due to the same wandb version.

    Parameters
    ----------
    cfg : [type]
        configuration used for training
    replace_wandb_only : bool, optional
        whether to replace wandb setting only or replace all settings, by
        default False
    """
    path_cwd = Path(Path(".").resolve())
    logger.info(f"Working directory: {path_cwd}")

    exp = Experiment(path_cwd)

    if exp.last_ckpt_path.exists():

        # replace cfg with previous configuration
        if not replace_wandb_only:
            logger.warning(f"Replace the whole config with: {exp.config_path}.")
            cfg = exp.config

        # else, only replace wandb related config
        elif element_exists(exp.config.logging.wandb) and element_exists(
            cfg.logging.wandb
        ):
            logger.info("Will use previous wandb name and version.")
            pre_wandb_name = exp.config.logging.wandb.name
            pre_wandb_version = exp.config.logging.wandb.version

            logger.info(f"Previous wandb name: {pre_wandb_name}")
            logger.info(f"Previous wandb version: {pre_wandb_version}")

            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.logging.wandb.name = pre_wandb_name
                cfg.logging.wandb.version = pre_wandb_version

        logger.info(f"Find previous checkpoint: {exp.last_ckpt_path}")

        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.resume_ckpt = str(exp.last_ckpt_path)

    return cfg


def prepare_trainer_config(cfg, logging=True):
    cfg_trainer = dict(cfg.pl_trainer)

    if logging and "logging" in cfg:
        loggers = []
        for _, cfg_log in cfg.logging.items():
            loggers.append(instantiate(cfg_log))
        cfg_trainer["logger"] = loggers

    if cfg.callbacks:
        callbacks = []
        for _, cfg_callback in cfg.callbacks.items():
            callbacks.append(instantiate(cfg_callback))
        cfg_trainer["callbacks"] = callbacks

    if cfg_trainer["strategy"] == "ddp":
        from pytorch_lightning.strategies.ddp import DDPStrategy

        cfg_trainer["strategy"] = DDPStrategy(find_unused_parameters=False)

    return cfg_trainer
