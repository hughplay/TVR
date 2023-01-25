import logging
from typing import Dict, Union

import torch
from torch import nn

from src.utils.datatool import write_json, write_jsonlines

from .components.evaluator import BasicEvaluator, EventEvaluator, ViewEvaluator
from .components.recorder import BasicRecorder, EventRecorder, ViewRecorder
from .loss import BasicLoss, ReinforceLoss

logger = logging.getLogger(__name__)


class ReasonCriterion(nn.Module):
    def __init__(
        self,
        loss: Union[BasicLoss, ReinforceLoss],
        evaluator: Union[BasicEvaluator, EventEvaluator, ViewEvaluator],
        recorder: Union[BasicRecorder, EventRecorder, ViewRecorder],
        compute_metrics_during_training: bool = False,
    ):
        super().__init__()

        self.loss = loss
        self.evaluator = evaluator
        self.recorder = recorder
        self.compute_metrics_during_training = compute_metrics_during_training

        self.OBJ_PAD = self.evaluator.t.OBJ_PAD
        self.PAIR_PAD = self.evaluator.t.PAIR_PAD

    def reset(self):
        self.recorder.reset()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        record_detail: bool = False,
    ):
        metrics = {}
        if self.compute_metrics_during_training:
            record_detail = True
        if not self.training or self.compute_metrics_during_training:
            metrics = self.evaluator.evaluate(
                **outputs, record_detail=record_detail
            )
        if compute_loss:
            loss_dict = self.loss(
                **outputs,
                metrics=metrics["detail"] if record_detail else None,
                obj_ignore_index=self.OBJ_PAD,
                pair_ignore_index=self.PAIR_PAD
            )
            metrics = {**metrics, **loss_dict}

        data = {**outputs, **metrics}
        if record_detail and "detail" in data:
            data["detail"]["sample_id"] = outputs["sample_id"].tolist()
        self.recorder.update(data)
        return metrics

    def compute(self, verbose=False):
        metrics = self.recorder.compute()
        if verbose:
            logger.info(self.recorder.summary(metrics))
        return metrics

    def save(
        self,
        path_detail: str = "detail.jsonl",
        path_summary: str = "summary.json",
    ):
        write_jsonlines(path_detail, self.recorder.records)
        write_json(path_summary, self.recorder.compute())
