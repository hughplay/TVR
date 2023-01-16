import os
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from omegaconf import OmegaConf

from src.dataset.trance import TRANCEAPI, ProblemName, SplitName
from src.utils.datatool import read_json, read_jsonlines


def get_root_variable(name, default):
    root = Path(os.getenv(name, default))
    if not root.is_dir():
        raise ValueError(f"{name}={root} is not a valid directory")
    else:
        print(f"{name}={root}")
    return root


# DATA_ROOT is the path to the data directory
data_root = get_root_variable("DATA_ROOT", "/data/trance/")
# EXP_ROOT is the path to the experiment log directory
log_root = get_root_variable("LOG_ROOT", "/log/exp/tvr/")

data_file = data_root / "data.h5"
trance_api = TRANCEAPI(data_file=data_file)

exp_ids_cache = {}
exp_detail_cache = {}


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "dataset": "TRANCE",
        "summary": trance_api.summary,
    }


@app.get("/image/{name}")
async def get_image(name: str):
    image_path = Path(data_root) / "image" / name
    return FileResponse(image_path)


@app.get("/sample/{problem}/{split}/{idx}")
async def sample(problem: ProblemName, split: SplitName, idx: int):
    try:
        meta = trance_api.get(problem, split, idx)
    except Exception as e:
        return {"error": str(e)}
    return meta


@app.get("/explist")
async def exps():
    exp_names = []
    for exp_root in sorted(Path(log_root).glob("*")):
        if (exp_root / "detail.jsonl").exists():
            config = OmegaConf.load(exp_root / "config.yaml")
            exp_name = config.name
            exp_id = exp_root.name
            exp_time = exp_id.split(".")[-1]
            i = 1
            while True:
                if (
                    exp_name in exp_ids_cache
                    and exp_ids_cache[exp_name]["id"] != exp_id
                ):
                    exp_name = f"{config.name}_{i}"
                    i += 1
                else:
                    break
            exp_ids_cache[exp_name] = {
                "id": exp_id,
                "time": exp_time,
                "root": str(exp_root),
                "problem": config.dataset.dataset_cfg.problems.test,
            }
            exp_names.append((exp_name, exp_time))

    exp_names = [
        x[0] for x in sorted(exp_names, key=lambda x: x[1], reverse=True)
    ]

    return exp_names


def get_exp_info(exp_root):
    exp_root = Path(exp_root)
    info = {}
    if exp_root.exists():
        summary_path = (
            exp_root / "wandb" / "latest-run" / "files" / "wandb-summary.json"
        )
        if summary_path.exists():
            info["result"] = read_json(summary_path)

    detail_path = exp_root / "detail.jsonl"
    if detail_path.exists():
        info["detail"] = read_jsonlines(detail_path)

    return info


@app.post("/exps")
async def exp(exp_names: List[str], idx: int = 0):

    print(exp_names)

    problem = None
    split = "test"
    results = {"gt": {}, "preds": [], "metrics": []}

    for exp_name in exp_names:
        if exp_name not in exp_detail_cache:
            if exp_name in exp_ids_cache:
                exp_root = exp_ids_cache[exp_name]["root"]
                exp_detail_cache[exp_name] = get_exp_info(exp_root)
            else:
                return {"error": "No such experiment"}

        if problem is None:
            problem = exp_ids_cache[exp_name]["problem"]
        elif problem != exp_ids_cache[exp_name]["problem"]:
            return {"error": "Inconsistent problems"}

        if idx >= len(exp_detail_cache[exp_name]["detail"]) or idx < 0:
            return {"error": "Out of range"}

        # prepare the prediction
        exp_res = exp_detail_cache[exp_name]["detail"][idx].copy()
        exp_res["name"] = exp_name
        exp_res["pred"] = trance_api.parse(exp_res["pred"])
        exp_res["target"] = trance_api.parse(exp_res["target"])

        results["preds"].append(exp_res)

        # prepare the metrics
        exp_metrics = exp_detail_cache[exp_name]["result"].copy()
        exp_metrics["name"] = exp_name

        results["metrics"].append(exp_metrics)

    if problem:
        results["gt"] = trance_api.get(problem, split, idx)
        results["problem"] = problem

    return results


if __name__ == "__main__":
    uvicorn.run("app", port=8000, log_level="info")
