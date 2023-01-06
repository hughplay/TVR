from typing import List

import numpy as np
import torch
from torch import nn

from src.dataset.components.vectorizer import Vectorizer


class Visibility:
    invisible = 0
    visible = 1
    invalid = 2


class BasicEvaluator(nn.Module):
    def __init__(
        self,
        values_json="trance/resource/values.json",
        properties_json="trance/resource/properties.json",
    ):
        super().__init__()

        self.t = Vectorizer(
            values_json=values_json, properties_json=properties_json
        )
        self.pair2attr = torch.tensor(
            [
                self.t.attrs.index(pair.split(self.t.SPLIT)[0])
                for pair in self.t.pairs
            ]
        )

    def evaluate(
        self,
        obj_choice: torch.Tensor = None,
        pair_choice: torch.Tensor = None,
        obj_target: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        options: torch.Tensor = None,
        record_detail: bool = False,
        **kwargs,
    ):
        obj_pred_idx = obj_choice.argmax(dim=1, keepdim=True)
        obj_target_idx = obj_target.view_as(obj_pred_idx)
        obj_correct = obj_pred_idx.eq(obj_target_idx)

        pair_pred_vec, pair_pred_idx = pair_choice.max(dim=1, keepdim=True)
        pair_pred_vec = pair_choice.eq(pair_pred_vec)
        pair_target_idx = pair_target.view_as(pair_pred_idx)
        pair_correct = (options * pair_pred_vec).sum(dim=1, keepdim=True) >= 1

        attr_pred_idx = self.pair2attr.to(pair_pred_idx)[pair_pred_idx]
        attr_target_idx = self.pair2attr.to(pair_target_idx)[pair_target_idx]
        attr_correct = attr_pred_idx.eq(attr_target_idx)

        correct = obj_correct * pair_correct

        n_sample = obj_choice.shape[0]
        info = {
            "n_sample": n_sample,
            "acc": correct.sum().item(),
            "acc_obj": obj_correct.sum().item(),
            "acc_attr": attr_correct.sum().item(),
            "acc_pair": pair_correct.sum().item(),
        }
        if record_detail:
            info["detail"] = {
                "correct": correct.squeeze().tolist(),
                "obj_correct": obj_correct.squeeze().tolist(),
                "pair_correct": pair_correct.squeeze().tolist(),
                "obj_pred": obj_pred_idx.squeeze().tolist(),
                "obj_target": obj_target_idx.squeeze().tolist(),
                "pair_pred": pair_pred_idx.squeeze().tolist(),
                "pair_target": pair_target_idx.squeeze().tolist(),
                "attr_pred": attr_pred_idx.squeeze().tolist(),
                "attr_target": attr_target_idx.squeeze().tolist(),
            }

        return info


class EventEvaluator(nn.Module):
    def __init__(
        self,
        values_json="trance/resource/values.json",
        properties_json="trance/resource/properties.json",
        valid_attrs=["position", "shape", "size", "color", "material"],
    ):
        super().__init__()

        self.t = Vectorizer(
            values_json=values_json,
            properties_json=properties_json,
            valid_attrs=valid_attrs,
        )
        self.tensor_size = torch.tensor([self.t.coord[s] for s in self.t.size])

        self.EPSILON = 1e-4

    def evaluate(
        self,
        obj_choice: torch.Tensor = None,
        pair_choice: torch.Tensor = None,
        init_desc: torch.Tensor = None,
        fin_desc: torch.Tensor = None,
        obj_target: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        record_detail: bool = False,
        **kwargs,
    ):
        preds = (obj_choice.argmax(dim=2), pair_choice.argmax(dim=2))
        targets = (obj_target, pair_target)

        res = self.eval_tensor_results(
            preds, init_desc, fin_desc, targets, keep_tensor=True
        )

        n_sample = obj_choice.shape[0]
        info = {
            "n_sample": n_sample,
            "acc": res["correct"].sum().item(),
            "loose_acc": res["loose_correct"].sum().item(),
            "avg_dist": res["dist"].sum().item(),
            "avg_norm_dist": res["norm_dist"].sum().item(),
            "avg_step_diff": (res["pred_step"] - res["target_step"])
            .sum()
            .item(),
        }

        if record_detail:
            res = {
                k: v.squeeze().tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()
            }
            info["detail"] = res

        return info

    def eval_text_results(self, predictions, samples, keep_tensor=False):
        init_mat_batch, fin_mat_batch = [], []
        targets, preds = [], []
        for sample, prediction in zip(samples, predictions):
            init_mat_batch.append(
                self.t.desc2mat(sample["states"][0]["objects"])
            )
            fin_mat_batch.append(
                self.t.desc2mat(sample["states"][-1]["objects"])
            )

            targets.append(self.t.multitrans2vec(sample["transformations"]))
            preds.append(self.t.multitrans2vec(prediction))
        init_mat_batch = torch.from_numpy(np.stack(init_mat_batch))
        fin_mat_batch = torch.from_numpy(np.stack(fin_mat_batch))
        return self.eval_tensor_results(
            preds, init_mat_batch, fin_mat_batch, targets, keep_tensor
        )

    def eval_text_result(self, sample, prediction):
        res = self.eval_text_results([prediction], [sample])
        return {k: v[0] for k, v in res.items()}

    def eval_tensor_results(
        self, preds, init_mat_batch, fin_mat_batch, targets, keep_tensor=False
    ):
        B = init_mat_batch.shape[0]
        device = init_mat_batch.device

        if type(preds[0]) is torch.Tensor:
            preds = self.unpack_batch(*preds)
        if type(targets[0]) is torch.Tensor:
            targets = self.unpack_batch(*targets)

        error_overlap = torch.full((B,), False, dtype=torch.bool).to(device)
        error_out = torch.full((B,), False, dtype=torch.bool).to(device)

        for i, (init_mat, pred) in enumerate(zip(init_mat_batch, preds)):
            error_overlap[i], error_out[i] = self.transform(init_mat, pred)

        # init_mat_batch has been transformed
        dist = self.compute_diff(init_mat_batch, fin_mat_batch)
        loose_correct = dist == 0
        pred_step = torch.tensor([len(p) for p in preds]).to(device)
        target_step = torch.tensor([len(t) for t in targets]).to(device)

        res = {
            "dist": dist,
            "norm_dist": 1.0 * dist / target_step,
            "loose_correct": loose_correct,
            "err_overlap": error_overlap,
            "err_invalid_position": error_out,
            "correct": loose_correct & ~error_overlap & ~error_out,
            "pred": preds,
            "target": targets,
            "pred_step": pred_step,
            "target_step": target_step,
        }

        if not keep_tensor:
            res = {
                k: v.tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()
            }

        return res

    def unpack_batch(self, obj_batch, pair_batch):
        res = []
        for objs, pairs in zip(obj_batch, pair_batch):
            sample = []
            for obj, pair in zip(objs, pairs):
                if obj >= self.t.OBJ_EOS or pair >= self.t.PAIR_EOS:
                    break
                sample.append((obj.item(), pair.item()))
            res.append(sample)
        return res

    def compute_diff(self, pred_fin_mat, target_fin_mat):
        diff = torch.abs(pred_fin_mat - target_fin_mat) > self.EPSILON

        pos_pred = self.t.restore_position(
            pred_fin_mat[
                ..., self.t.feat_start["position"] : self.t.feat_end["position"]
            ]
        )
        pos_target = self.t.restore_position(
            target_fin_mat[
                ..., self.t.feat_start["position"] : self.t.feat_end["position"]
            ]
        )
        pos_equal = (
            torch.sum(torch.abs(pos_pred - pos_target) < self.EPSILON, dim=2)
            == pos_pred.shape[2]
        )

        vis_pred = self.b_is_visible(pos_pred)
        vis_target = self.b_is_visible(pos_target)

        pos_equal = ((vis_target == Visibility.visible) & pos_equal) | (
            (vis_target == Visibility.invisible)
            & (vis_pred == Visibility.invisible)
        )
        diff[
            ..., self.t.feat_start["position"] : self.t.feat_end["position"]
        ] = ~pos_equal[..., None]

        n_diff = torch.sum(diff, dim=(1, 2)) / 2

        return n_diff

    def transform(self, init_mat, pred):
        error_overlap = error_out = False
        for obj_idx, pair_idx in pred:
            e_overlap, e_out = self.transform_step(init_mat, obj_idx, pair_idx)
            error_overlap |= e_overlap
            error_out |= e_out
        return error_overlap, error_out

    def transform_step(self, objs, obj_idx, pair_idx):
        attr, value = self.t.sep_pair(pair_idx)
        error_overlap = error_out = False
        obj = objs[obj_idx]

        func = getattr(self, "t_{}".format(attr))
        func(obj, value)

        if attr == "position":
            after = self.is_visible(
                obj[self.t.feat_start["position"] : self.t.feat_end["position"]]
            )
            if after is Visibility.invalid:
                error_out = True

        if attr in ["position", "size"]:
            error_overlap = self.is_overlap(objs, obj_idx)

        return error_overlap, error_out

    def t_position(self, obj, target):
        direction, step = target
        v_direction = self.t.properties["position"]["direction"][direction]

        obj[self.t.feat_start["position"] : self.t.feat_end["position"]] += (
            torch.tensor(v_direction).to(obj)
            * step
            * self.t.coord["step"]
            / (self.t.pos_max - self.t.pos_min)
        )

    def t_material(self, obj, material):
        obj[self.t.feat_start["material"] : self.t.feat_end["material"]] = 0
        obj[self.t.feat_start["material"] + self.t.material.index(material)] = 1

    def t_color(self, obj, color):
        obj[self.t.feat_start["color"] : self.t.feat_end["color"]] = 0
        obj[self.t.feat_start["color"] + self.t.color.index(color)] = 1

    def t_shape(self, obj, shape):
        obj[self.t.feat_start["shape"] : self.t.feat_end["shape"]] = 0
        obj[self.t.feat_start["shape"] + self.t.shape.index(shape)] = 1

    def t_size(self, obj, size):
        obj[self.t.feat_start["size"] : self.t.feat_end["size"]] = 0
        obj[self.t.feat_start["size"] + self.t.size.index(size)] = 1

    def is_overlap(self, state_mat, obj_idx):
        pos = state_mat[
            :, self.t.feat_start["position"] : self.t.feat_end["position"]
        ]
        size_idx = torch.argmax(
            state_mat[:, self.t.feat_start["size"] : self.t.feat_end["size"]],
            dim=1,
        )
        size = self.tensor_size.to(size_idx)[size_idx]

        obj_size = size[obj_idx]
        obj_pos = pos[obj_idx]

        dist = torch.norm(pos - obj_pos, dim=1) * (
            self.t.pos_max - self.t.pos_min
        )
        min_dist = size + obj_size + self.t.coord["min_gap"]

        n_violate = torch.sum(dist + self.EPSILON < min_dist.to(dist))
        res = (n_violate > 1).item()

        return res

    def is_visible(self, pos):
        pos = self.t.restore_position(pos)
        x = pos[..., 0]
        y = pos[..., 1]
        if (
            self.t.coord["x_min"] <= x <= self.t.coord["x_max"]
            and self.t.coord["y_min"] <= y <= self.t.coord["y_max"]
        ):
            if (
                self.t.coord["vis_x_min"] <= x <= self.t.coord["vis_x_max"]
                and self.t.coord["vis_y_min"] <= y <= self.t.coord["vis_y_max"]
            ):
                return Visibility.visible
            else:
                return Visibility.invisible
        else:
            return Visibility.invalid

    def b_is_visible(self, pos):
        x = pos[..., 0]
        y = pos[..., 1]

        state = torch.full(
            pos.shape[:-1], Visibility.invisible, dtype=torch.uint8
        ).to(pos.device)

        invalid_x = (x > self.t.coord["x_max"]) | (x < self.t.coord["x_min"])
        invalid_y = (y > self.t.coord["y_max"]) | (y < self.t.coord["y_min"])

        vis_x = (x >= self.t.coord["vis_x_min"]) & (
            x <= self.t.coord["vis_x_max"]
        )
        vis_y = (y >= self.t.coord["vis_y_min"]) & (
            y <= self.t.coord["vis_y_max"]
        )

        state[invalid_x | invalid_y] = Visibility.invalid
        state[vis_x & vis_y] = Visibility.visible

        return state


class ViewEvaluator(EventEvaluator):
    def evaluate(
        self,
        obj_choice: torch.Tensor = None,
        pair_choice: torch.Tensor = None,
        init_desc: torch.Tensor = None,
        fin_desc: torch.Tensor = None,
        obj_target: torch.Tensor = None,
        pair_target: torch.Tensor = None,
        view_idx: torch.Tensor = None,
        final_views: List[str] = [
            "Camera_Center",
            "Camera_Left",
            "Camera_Right",
        ],
        record_detail: bool = False,
        **kwargs,
    ):

        preds = (obj_choice.argmax(dim=2), pair_choice.argmax(dim=2))
        targets = (obj_target, pair_target)

        res = self.eval_tensor_results(
            preds, init_desc, fin_desc, targets, keep_tensor=True
        )

        n_sample = obj_choice.shape[0]
        info = {
            "n_sample": n_sample,
            "final_views": [],
            "acc": res["correct"].sum().item(),
            "loose_acc": res["loose_correct"].sum().item(),
            "avg_dist": res["dist"].sum().item(),
            "avg_norm_dist": res["norm_dist"].sum().item(),
            "avg_step_diff": (res["pred_step"] - res["target_step"])
            .sum()
            .item(),
        }

        for i, view in enumerate(final_views):
            view = view.split("_")[-1].lower()
            info["final_views"].append(view)
            selected = view_idx == i
            info["{}/acc".format(view)] = res["correct"][selected].sum().item()
            info["{}/loose_acc".format(view)] = (
                res["loose_correct"][selected].sum().item()
            )
            info["{}/avg_dist".format(view)] = (
                res["dist"][selected].sum().item()
            )
            info["{}/avg_norm_dist".format(view)] = (
                res["norm_dist"][selected].sum().item()
            )
            info["{}/avg_step_diff".format(view)] = (
                (res["pred_step"][selected] - res["target_step"][selected])
                .sum()
                .item()
            )
            info["n_{}_sample".format(view)] = selected.sum().item()

        if record_detail:
            res = {
                k: v.squeeze().tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()
            }
            info["detail"] = res
            info["detail"]["view_idx"] = view_idx.squeeze().tolist()

        return info
