from collections import OrderedDict, defaultdict

import numpy as np
from torch import nn


class BasicRecorder(nn.Module):

    EPSILON = 1e-8

    def __init__(self):
        super().__init__()
        self.problem = "basic"
        self.reset()

    def compute(self):
        return {
            "loss": self.get_average("loss"),
            "loss_obj": self.get_average("loss_obj"),
            "loss_pair": self.get_average("loss_pair"),
            "acc": self.get_average("acc"),
            "acc_obj": self.get_average("acc_obj"),
            "acc_pair": self.get_average("acc_pair"),
            "acc_attr": self.get_average("acc_attr"),
        }

    @property
    def records(self):
        record_list = []
        # transform detail to list
        key = list(self.detail.keys())[0]
        for i in range(len(self.detail[key])):
            record = {}
            for k, v in self.detail.items():
                record[k] = v[i]
            record_list.append(record)
        return record_list

    def summary(self, metrics=None):
        if metrics is None:
            metrics = self.compute()
        return (
            "loss: {loss:.4f} ({loss_obj:.4f}; {loss_pair:.4f}),"
            " acc: {acc:.4f} ({acc_obj:.4f}, {acc_attr:.4f}, "
            "{acc_pair:.4f})"
        ).format(**metrics)

    def get_average(self, key):
        return sum(self.metrics[key]) / (
            sum(self.metrics["n_sample"]) + self.EPSILON
        )

    def reset(self):
        # self.t_start = time.time()
        self.metrics = defaultdict(list)
        self.detail = defaultdict(list)

    def update(self, info):
        for key, val in info.items():
            if key == "detail":
                for k, v in val.items():
                    self.detail[k].extend(v)
            else:
                self.metrics[key].append(val)

    def forward(self, **kwargs):
        self.update(kwargs)
        return kwargs


class EventRecorder(BasicRecorder):
    def __init__(self, enable_step_result=True):
        super().__init__()
        self.enable_step_result = enable_step_result
        self.problem = "event"

    def compute(self):
        state = OrderedDict(
            {
                "loss": self.get_average("loss"),
                "loss_obj": self.get_average("loss_obj"),
                "loss_pair": self.get_average("loss_pair"),
                "acc": self.get_average("acc"),
                "loose_acc": self.get_average("loose_acc"),
                "avg_dist": self.get_average("avg_dist"),
                "avg_norm_dist": self.get_average("avg_norm_dist"),
                "avg_step_diff": self.get_average("avg_step_diff"),
            }
        )
        if self.enable_step_result:
            steps = np.unique(np.array(self.detail["target_step"])).tolist()
            for step in steps:
                idx = np.array(self.detail["target_step"]) == step
                key_prefix = f"step_{step}"
                state[f"{key_prefix}/acc"] = float(
                    np.mean(np.array(self.detail["correct"])[idx])
                )
                state[f"{key_prefix}/loose_acc"] = float(
                    np.mean(np.array(self.detail["loose_correct"])[idx])
                )
                state[f"{key_prefix}/avg_dist"] = float(
                    np.mean(np.array(self.detail["dist"])[idx])
                )
                state[f"{key_prefix}/avg_norm_dist"] = float(
                    np.mean(np.array(self.detail["norm_dist"])[idx])
                )
        return dict(state)

    def summary(self, metrics=None):
        if metrics is None:
            metrics = self.compute()
        return (
            "loss: {loss:.4f}, acc: {acc:.4f} ({loose_acc:.4f}), "
            "AD: {avg_dist:.4f}, AND: {avg_norm_dist:.4f}, "
            "step_diff: {avg_step_diff:.2f}".format(**metrics)
        )


class ViewRecorder(EventRecorder):
    def compute(self):
        state = OrderedDict(
            {
                "loss": self.get_average("loss"),
                "loss_obj": self.get_average("loss_obj"),
                "loss_pair": self.get_average("loss_pair"),
                "acc": self.get_average("acc"),
                "loose_acc": self.get_average("loose_acc"),
                "avg_dist": self.get_average("avg_dist"),
                "avg_norm_dist": self.get_average("avg_norm_dist"),
                "avg_step_diff": self.get_average("avg_step_diff"),
            }
        )
        for view in self.metrics["final_views"]:
            for key in [
                "acc",
                "loose_acc",
                "avg_dist",
                "avg_norm_dist",
                "avg_step_diff",
            ]:
                view_metric = "{}/{}".format(view, key)
                state[view_metric] = sum(self.metrics[view_metric]) / sum(
                    self.metrics["n_{}_sample".format(view)]
                )

        if self.enable_step_result:
            steps = np.unique(np.array(self.detail["target_step"])).tolist()

            for step in steps:
                step_prefix = f"step_{step}"
                idx = np.array(self.detail["target_step"]) == step
                state[f"{step_prefix}/acc"] = float(
                    np.mean(np.array(self.detail["correct"])[idx])
                )
                state[f"{step_prefix}/loose_acc"] = float(
                    np.mean(np.array(self.detail["loose_correct"])[idx])
                )
                state[f"{step_prefix}/avg_dist"] = float(
                    np.mean(np.array(self.detail["dist"])[idx])
                )
                state[f"{step_prefix}/avg_norm_dist"] = float(
                    np.mean(np.array(self.detail["norm_dist"])[idx])
                )
                state[f"{step_prefix}/avg_step_diff"] = float(
                    np.mean(
                        np.array(self.detail["pred_step"])[idx]
                        - np.array(self.detail["target_step"])[idx]
                    )
                )

                for i, view in enumerate(self.metrics["final_views"]):
                    view_prefix = f"{step_prefix}/{view}"
                    v_idx = idx * (np.array(self.detail["view_idx"]) == i)
                    state[f"{view_prefix}/acc"] = float(
                        np.mean(np.array(self.detail["correct"])[v_idx])
                    )
                    state[f"{view_prefix}/loose_acc"] = float(
                        np.mean(np.array(self.detail["loose_correct"])[v_idx])
                    )
                    state[f"{view_prefix}/avg_dist"] = float(
                        np.mean(np.array(self.detail["dist"])[v_idx])
                    )
                    state[f"{view_prefix}/avg_norm_dist"] = float(
                        np.mean(np.array(self.detail["norm_dist"])[v_idx])
                    )
                    state[f"{view_prefix}/avg_step_diff"] = float(
                        np.mean(
                            np.array(self.detail["pred_step"])[v_idx]
                            - np.array(self.detail["target_step"])[v_idx]
                        )
                    )
        return dict(state)

    def update(self, info):
        for key, val in info.items():
            if key == "detail":
                for k, v in val.items():
                    self.detail[k].extend(v)
            elif key == "final_views":
                self.metrics[key] = val
            else:
                self.metrics[key].append(val)

    def summary(self, metrics=None):
        if metrics is None:
            metrics = self.compute()
        return (
            "loss: {loss:.4f}, acc: {acc:.4f} ({loose_acc:.4f}), "
            "AD: {avg_dist:.4f}, AND: {avg_norm_dist:.4f}, "
            "step_diff: {avg_step_diff:.2f}".format(**metrics)
        )


class ReinforceEventRecorder(BasicRecorder):
    def compute(self):
        state = OrderedDict(
            {
                "loss": self.get_average("loss"),
                "loss_obj": self.get_average("loss_obj"),
                "loss_pair": self.get_average("loss_pair"),
                "reward": self.get_average("reward"),
                "acc": self.get_average("acc"),
                "loose_acc": self.get_average("loose_acc"),
                "avg_dist": self.get_average("avg_dist"),
                "avg_norm_dist": self.get_average("avg_norm_dist"),
                "avg_step_diff": self.get_average("avg_step_diff"),
            }
        )
        if self.enable_step_result:
            steps = np.unique(np.array(self.detail["target_step"])).tolist()
            for step in steps:
                idx = np.array(self.detail["target_step"]) == step
                key_prefix = f"step_{step}"
                state[f"{key_prefix}/reward"] = float(
                    np.mean(np.array(self.detail["reward"])[idx])
                )
                state[f"{key_prefix}/acc"] = float(
                    np.mean(np.array(self.detail["correct"])[idx])
                )
                state[f"{key_prefix}/loose_acc"] = float(
                    np.mean(np.array(self.detail["loose_correct"])[idx])
                )
                state[f"{key_prefix}/avg_dist"] = float(
                    np.mean(np.array(self.detail["dist"])[idx])
                )
                state[f"{key_prefix}/avg_norm_dist"] = float(
                    np.mean(np.array(self.detail["norm_dist"])[idx])
                )
        return dict(state)

    def summary(self, metrics=None):
        if metrics is None:
            metrics = self.compute()
        return (
            "loss: {loss:.4f}, reward: {reward:.4f}, acc: {acc:.4f} ({loose_acc:.4f}), "
            "AD: {avg_dist:.4f}, AND: {avg_norm_dist:.4f}, "
            "step_diff: {avg_step_diff:.2f}".format(**metrics)
        )
