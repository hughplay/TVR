import logging
from typing import Any, Dict, List

from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import get_model_size_mb

logger = logging.getLogger(__name__)


class ReasonLitModule(LightningModule):
    """Example of LightningModule for MNIST classification. A LightningModule
    organizes your PyTorch code into 5 sections:

        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(cfg)

        # initialize the model from configuration
        self.model = instantiate(self.hparams.model)

        # initialize the criterion from configuration
        self.criterion = instantiate(self.hparams.criterion)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = instantiate(self.hparams.optim, self.parameters())
        scheduler = instantiate(
            self._set_num_training_steps(self.hparams.scheduler), optimizer
        )
        # torch's scheduler is epoch-based, but transformers' is step-based
        interval = (
            "step"
            if self.hparams.scheduler._target_.startswith("transformers")
            else "epoch"
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            logger.info("Computing number of training steps...")
            scheduler_cfg[
                "num_training_steps"
            ] = self.trainer.estimated_stepping_batches

            if self.global_rank == 0:
                logger.info(
                    f"Training steps: {scheduler_cfg['num_training_steps']}"
                )
        return scheduler_cfg

    def step(
        self,
        batch: Dict[str, Any],
        compute_loss: bool = True,
        record_detail: bool = False,
    ):
        inputs = {
            "init": batch["init"],
            "fin": batch["fin"],
            "init_desc": batch["init_desc"],
            "obj_target_vec": batch["obj_target_vec"],
            "pair_target": batch["pair_target"],
        }

        outputs = self.model(**inputs)
        outputs.update(
            {
                "obj_target": batch["obj_target"],
                "pair_target": batch["pair_target"],
                "init_desc": batch["init_desc"],
                "sample_id": batch["sample_id"],
            }
        )

        for key in ["options", "fin_desc", "view_idx"]:
            if key in batch:
                outputs.update({key: batch[key]})

        outputs = self.criterion(
            outputs, compute_loss=compute_loss, record_detail=record_detail
        )
        return outputs

    def on_train_start(self):
        self.log(
            "model_size/total",
            get_model_size_mb(self.model),
            rank_zero_only=True,
            logger=True,
        )
        self.criterion.reset()

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, compute_loss=True)

        metrics = self.criterion.compute()
        for name, value in metrics.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
            )

        return outputs["loss"]

    def on_validation_start(self):
        self.criterion.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        self.step(batch, compute_loss=False)

    def validation_epoch_end(self, outputs: List[Any]):
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
            )

    def on_test_start(self):
        self.criterion.reset()

    def test_step(self, batch: Any, batch_idx: int):
        self.step(batch, compute_loss=False, record_detail=True)

    def test_epoch_end(self, outputs: List[Any]):
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"test/{name}",
                value,
                on_epoch=True,
                prog_bar=True,
            )
        self.criterion.save()
