import logging

import dotenv
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("problem", ["basic", "event", "view"])
def test_datamodule(problem):

    batch_size = 2

    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"dataset={problem}",
                f"dataset.batch_size={batch_size}",
            ],
        )

    datamodule = instantiate(cfg.dataset)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader
    assert val_loader
    assert test_loader

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    batch = next(iter(train_loader))

    assert len(batch["init"]) == batch_size
    assert len(batch["fin"]) == batch_size
    assert len(batch["init_desc"]) == batch_size
    assert len(batch["target"]) == batch_size
    assert len(batch["obj_target_vec"]) == batch_size

    if problem == "basic":
        assert len(batch["options"]) == batch_size
    else:
        assert len(batch["view"]) == batch_size
        assert len(batch["fin_desc"]) == batch_size
