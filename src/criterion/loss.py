from typing import Dict

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn


class BasicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        obj_choice: torch.Tensor = None,
        pair_choice: torch.Tensor = None,
        obj_target: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        obj_ignore_index: int = -100,
        pair_ignore_index: int = -100,
        **kwargs
    ):
        assert obj_choice.dim() == pair_choice.dim()

        if obj_choice.dim() == 3:
            obj_choice = obj_choice.view(-1, obj_choice.shape[-1])
            obj_target = obj_target.view(-1)
            pair_choice = pair_choice.view(-1, pair_choice.shape[-1])
            pair_target = pair_target.view(-1)

        obj_result = F.nll_loss(
            obj_choice, obj_target, ignore_index=obj_ignore_index
        )
        pair_result = F.nll_loss(
            pair_choice, pair_target, ignore_index=pair_ignore_index
        )

        loss = obj_result + pair_result

        result = {
            "loss": loss,
            "obj_loss": obj_result,
            "pair_loss": pair_result,
        }
        return result


class ReinforceLoss(nn.Module):
    def __init__(
        self,
        reward_type: str = "acc_dist",
    ):
        super().__init__()

        self.reward_type = reward_type

    def get_reward(self, metrics: Dict[str, torch.Tensor]):
        if self.reward_type == "acc":
            return torch.tensor(metrics["correct"]) + 1.0
        elif self.reward_type == "dist":
            reward = 2.0 - 1.0 * torch.tensor(metrics["norm_dist"])
        elif self.reward_type == "acc_dist":
            reward = (
                2.0
                + torch.tensor(metrics["correct"])
                - 1.0 * torch.tensor(metrics["norm_dist"])
            )
        else:
            raise NotImplementedError

        return reward.detach()

    def forward(
        self,
        obj_choice: torch.Tensor = None,
        pair_choice: torch.Tensor = None,
        obj_target: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        metrics: Dict[str, torch.Tensor] = None,
        obj_ignore_index: int = -100,
        pair_ignore_index: int = -100,
        **kwargs
    ):
        assert obj_choice.dim() == pair_choice.dim()
        reward = self.get_reward(metrics).to(obj_choice.device)

        if obj_choice.dim() == 3:
            B, L, _ = obj_choice.shape
            weight = repeat(reward, "B -> B L", B=B, L=L).reshape(-1)
            obj_choice = obj_choice.view(-1, obj_choice.shape[-1])
            obj_target = obj_target.view(-1)
            pair_choice = pair_choice.view(-1, pair_choice.shape[-1])
            pair_target = pair_target.view(-1)

        obj_result = torch.mean(
            F.nll_loss(
                obj_choice,
                obj_target,
                reduce="none",
                ignore_index=obj_ignore_index,
            )
            * weight
        )
        pair_result = torch.mean(
            F.nll_loss(
                pair_choice,
                pair_target,
                reduce="none",
                ignore_index=pair_ignore_index,
            )
            * weight
        )
        loss = obj_result + pair_result

        result = {
            "loss": loss,
            "obj_loss": obj_result,
            "pair_loss": pair_result,
            "reward": reward,
        }
        return result
