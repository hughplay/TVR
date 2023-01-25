import argparse
import json
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.criterion.components.evaluator import EventEvaluator  # noqa: E402
from src.utils.datatool import (  # noqa: E402
    read_json,
    read_jsonlines,
    write_json,
)

MODEL_NAMES = {
    "SingleSubtractCNN": "CNN$_{-}$-G",
    "SubtractCNN": "CNN$_{-}$-G",
    "SingleConcatCNN": r"CNN$_{\oplus}$-G",
    "ConcatCNN": r"CNN$_{\oplus}$-G",
    "BCNN": "BCNN-G",
    "SingleBCNN": "BCNN-G",
    "DUDA": "DUDA-G",
    "SingleDUDA": "DUDA-G",
    "SingleSubtractResNet": r"ResNet$_{-}$-G",
    "SubtractResNet": r"ResNet$_{-}$-G",
    "SingleConcatResNet": r"ResNet$_{\oplus}$-G",
    "ConcatResNet": r"ResNet$_{\oplus}$-G",
    "SubtractResNetFormer": r"ResNet$_{-}$-T",
    "ConcatResNetFormer": r"ResNet$_{\oplus}$-T",
}
UNIQUE_MODEL_NAMES = []
for model in MODEL_NAMES.values():
    if model not in UNIQUE_MODEL_NAMES:
        UNIQUE_MODEL_NAMES.append(model)
PATH_H5 = Path("/data/trance/data.h5")
PATH_ROOT = Path("/log/exp/tvr/")


def main(args):

    api = wandb.Api()

    # api.default_entity by default
    entity = api.default_entity if args.entity is None else args.entity

    # get runs from the project
    def filter_runs(filters=None, sort=None):
        runs = api.runs(f"{entity}/{args.project}", filters=filters)
        runs = [run for run in runs if ("test/acc" in run.summary)]
        if sort is not None:
            runs = sorted(runs, key=sort)
        print(f"Find {len(runs)} runs in {entity}/{args.project}")
        return runs

    latex_str = getattr(sys.modules[__name__], f"{args.table}_table")(
        filter_runs
    )
    print(latex_str)


def gen_latex(
    style, caption, position="ht", small=True, save_path=None, **kwargs
):
    print(r"\usepackage{booktabs}")
    print()
    latex_str = style.to_latex(
        caption=caption,
        hrules=True,
        position=position,
        position_float="centering",
        **kwargs,
    )
    if small:
        latex_str = latex_str.replace(
            "\\begin{tabular}", "\\begin{small}\n\\begin{tabular}"
        )
        latex_str = latex_str.replace(
            "\\end{tabular}", "\\end{tabular}\n\\end{small}"
        )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            f.write(latex_str)
    return latex_str


def baseline_table(filter_runs: callable):
    filters = {"tags": {"$in": ["base"]}}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/acc"])
    results = defaultdict(lambda: defaultdict(list))
    for run in runs[::-1]:
        if "test/acc" not in run.summary:
            continue
        model_name = MODEL_NAMES[run.config["model/_target_"].split(".")[-1]]
        problem = run.config["dataset/dataset_cfg/problems/test"].capitalize()
        if problem == "Basic":
            results[model_name][(problem, "ObjAcc")] = run.summary[
                "test/acc_obj"
            ]
            results[model_name][(problem, "AttrAcc")] = run.summary[
                "test/acc_attr"
            ]
            results[model_name][(problem, "ValAcc")] = run.summary[
                "test/acc_pair"
            ]
            results[model_name][(problem, "Acc")] = run.summary["test/acc"]
        else:
            results[model_name][(problem, "AD")] = run.summary["test/avg_dist"]
            results[model_name][(problem, "AND")] = run.summary[
                "test/avg_norm_dist"
            ]
            results[model_name][(problem, "LAcc")] = run.summary[
                "test/loose_acc"
            ]
            results[model_name][(problem, "Acc")] = run.summary["test/acc"]

    df = pd.DataFrame(results).T
    df.loc[:, r"$\Delta$Acc"] = (
        df.loc[:, ("View", "Acc")] - df.loc[:, ("Event", "Acc")]
    )
    df = df.reindex(UNIQUE_MODEL_NAMES)
    df = df.reset_index().rename(columns={"index": "Model"})

    style = df.style.highlight_max(
        axis=0,
        subset=df.columns[df.columns.get_level_values(1).str.contains("Acc")],
        props="textbf:--rwrap;",
    )
    style = style.highlight_max(
        axis=0,
        subset=df.columns[df.columns.get_level_values(0).str.contains("Acc")],
        props="textbf:--rwrap;",
    )
    style = style.highlight_min(
        axis=0,
        subset=df.columns[df.columns.get_level_values(1).str.contains("AD")],
        props="textbf:--rwrap;",
    )
    style = style.highlight_min(
        axis=0,
        subset=df.columns[df.columns.get_level_values(1).str.contains("AND")],
        props="textbf:--rwrap;",
    )

    style = style.format(precision=4).hide(axis="index")

    str_latex = gen_latex(
        style,
        "Model and human performance on Basic, Event, and View. "
        r"$\Delta$Acc is the accuracy difference between View and Event.",
        save_path="docs/tables/baseline.tex",
        label="tab:results",
        column_format="lrrrrrrrrrrrrr",
        position="t",
        small=True,
    )
    str_latex = str_latex.replace("nan", "-")
    str_latex = str_latex.replace("table", "table*")
    str_latex = str_latex.replace("{r}", "{c}")
    str_latex = str_latex.replace("Model &", r"\multirow{2}{*}{Model} &")
    str_latex = str_latex.replace(
        r"& $\Delta$Acc", r"& \multirow{2}{*}{$\Delta$Acc$\uparrow$}"
    )
    str_latex = str_latex.replace(" Acc", r" Acc$\uparrow$")
    str_latex = str_latex.replace("AD", r"AD$\downarrow$")
    str_latex = str_latex.replace("AND", r"AND$\downarrow$")
    lines = []
    for line in str_latex.split("\n"):
        if "bottomrule" in line:
            lines.append(r"\midrule")
            lines.append(
                r"Human & 1.0000 & 1.0000 & 1.0000 & 1.0000 & 0.3700 & 0.1200 & 0.8300 & 0.8300 & 0.3200 & 0.0986 & 0.8433 & 0.8433 & 0.0133 \\"
            )
        lines.append(line)
        if "multicolumn" in line:
            lines.append(
                r" \cmidrule[.5pt](lr){2-5}  \cmidrule[.5pt](lr){6-9}  \cmidrule[.5pt](lr){10-13}"
            )
        if r"\caption" in line:
            lines.append(r"\setlength{\tabcolsep}{0.45em}")
    str_latex = "\n".join(lines)

    return str_latex


def reinforce_table(filter_runs: callable):
    filters = {"tags": {"$in": ["reinforce_former"]}}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/acc"])

    results = defaultdict(list)
    primary_name = ""
    for run in runs:
        if "test/acc" not in run.summary:
            continue
        if "criterion/loss/reward_type" in run.config:
            reward = run.config["criterion/loss/reward_type"]
            if reward == "acc":
                name = r"\hspace{0.2em} + \textit{corr}"
            elif reward == "dist":
                name = r"\hspace{0.2em} + \textit{dist}"
            elif reward == "acc_dist":
                name = r"\hspace{0.2em} + \textit{corr \& dist}"
        else:
            name = MODEL_NAMES[run.config["model/_target_"].split(".")[-1]]
            primary_name = name
        results["Model"].append(name)
        results["AD"].append(run.summary["test/avg_dist"])
        results["AND"].append(run.summary["test/avg_norm_dist"])
        results["LAcc"].append(run.summary["test/loose_acc"])
        results["Acc"].append(run.summary["test/acc"])

    df = pd.DataFrame(results)

    max_metrics = [metric for metric in df.columns if "Acc" in metric]
    min_metrics = [
        metric for metric in df.columns if "AD" in metric or "AND" in metric
    ]

    style = df.style.highlight_max(
        axis=0,
        subset=max_metrics,
        props="textbf:--rwrap;",
    )
    style = style.highlight_min(
        axis=0,
        subset=min_metrics,
        props="textbf:--rwrap;",
    )
    style = style.format(precision=4).hide(axis="index")

    str_latex = gen_latex(
        style,
        f"Results of {primary_name} trained using"
        + r" REINFORCE~\cite{williams1992simple} with different rewards on Event.",
        label="tab:rl",
        save_path="docs/tables/diff.tex",
        position="t",
    )
    str_latex = str_latex.replace("Acc", r"Acc$\uparrow$")
    str_latex = str_latex.replace("AD", r"AD$\downarrow$")
    str_latex = str_latex.replace("AND", r"AND$\downarrow$")

    return str_latex


def compute_permutations(sample):
    trans = sample["transformations"]
    per = list(permutations(trans))
    e = EventEvaluator()
    res = e.eval_text_results(per, [sample] * len(per), True)
    res = {k: v * 1.0 for k, v in res.items() if type(v) is torch.Tensor}
    return torch.tensor(
        [
            res["err_overlap"].mean() > 0,
            res["err_invalid_position"].mean() > 0,
            res["loose_correct"].mean() < 1,
            res["correct"].mean() != 1,
            res["correct"].mean(),
        ]
    )


def get_idx_affected(path_cache="docs/order_sensitive_samples.json"):
    if not Path(path_cache).exists():
        with h5py.File(PATH_H5, "r") as f:
            keys = eval(f["test"]["keys"][()])
            samples = [json.loads(f["test"]["data"][key][()]) for key in keys]
        rate_event = []
        for sample in tqdm(samples, ncols=80):
            rate_event.append(compute_permutations(sample))
        rate_event = torch.stack(rate_event)
        idx = ((rate_event[:, 0] == 1) | (rate_event[:, 1] == 1)) & (
            rate_event[:, 2] == 0
        )
        write_json(path_cache, idx.tolist())

    idx = read_json(path_cache)
    return idx


def order_table(filter_runs: callable):
    filters = {
        "$and": [
            {"tags": {"$in": ["base"]}},
            {"tags": {"$in": ["event"]}},
        ],
    }
    runs = filter_runs(filters, sort=lambda run: run.summary["test/acc"])

    for run in runs:
        if "test/acc" not in run.summary:
            continue

    idx_affected = get_idx_affected()
    print(
        f"Proportion of order sensitive samples: {np.array(idx_affected).mean()*100:.2f}%"
    )
    idx_affected = torch.arange(len(idx_affected))[idx_affected]

    results = defaultdict(list)
    for run in runs:
        if "test/acc" not in run.summary:
            continue
        path_detail = PATH_ROOT / run.config["exp_id"] / "detail.jsonl"
        if path_detail.exists():
            samples = read_jsonlines(path_detail)
            samples = [samples[i] for i in idx_affected]
            loose_acc = np.array(
                [result["loose_correct"] for result in samples]
            ).mean()
            acc = np.array([result["correct"] for result in samples]).mean()
            eo = (loose_acc - acc) / loose_acc

            model_name = MODEL_NAMES[
                run.config["model/_target_"].split(".")[-1]
            ]
            results["Model"].append(model_name)
            results["LAcc"].append(loose_acc)
            results["Acc"].append(acc)
            results["EO"].append(eo)

    df = pd.DataFrame(results)
    df.reindex(columns=["Model", "LAcc", "Acc", "EO"])

    max_metrics = [metric for metric in df.columns if "Acc" in metric]
    min_metrics = [metric for metric in df.columns if "EO" in metric]

    style = df.style.highlight_max(
        axis=0,
        subset=max_metrics,
        props="textbf:--rwrap;",
    )
    style = style.highlight_min(
        axis=0,
        subset=min_metrics,
        props="textbf:--rwrap;",
    )
    style = style.format(precision=4).hide(axis="index")

    str_latex = gen_latex(
        style,
        r"Results on 7.8\% order sensitive samples from Event.",
        label="tab:order",
        save_path="docs/tables/diff.tex",
        position="t",
    )
    str_latex = str_latex.replace("Acc", r"Acc$\uparrow$")
    str_latex = str_latex.replace("EO", r"EO$\downarrow$")

    lines = []
    for line in str_latex.split("\n"):
        if "bottomrule" in line:
            lines.append(r"\midrule")
            lines.append(r"Human & 0.7273 & 0.7273 & 0.0000 \\")
        lines.append(line)
        if r"\midrule" in line:
            lines.append(r"Random (avg. of 100) & 1.0000 & 0.4992 & 0.5008 \\")
            lines.append(r"\midrule")
    str_latex = "\n".join(lines)

    return str_latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--table", default="main")
    parser.add_argument("--project", default="tvr")
    parser.add_argument("--caption", default="Model performance.")
    args = parser.parse_args()
    main(args)
