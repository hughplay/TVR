import io
import json
import pickle as pkl
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from evaluator import EventEvaluator
from flask import Flask, request, send_file
from flask_compress import Compress
from flask_cors import CORS
from PIL import Image
from ruamel.yaml import YAML

from dataset import Translator

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
Compress(app)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

translator = Translator()
info_dict = {}
sample_path = {}
human_test_dir = "human_test"

test_dict = {
    # 'basic': {
    #     'valid_attrs': ['position', 'shape', 'size', 'color', 'material'],
    #     'dir': '../data/TRANCE/single',
    #     'view_final': ['Camera_Center']
    # },
    # 'view': {
    #     'valid_attrs': ['position', 'shape', 'size', 'color', 'material'],
    #     'dir': '../data/TRANCE/single',
    #     'view_final': ['Camera_Left', 'Camera_Center', 'Camera_Right']
    # },
    # 'event': {
    #     'valid_attrs': ['position', 'shape', 'size', 'color', 'material'],
    #     'dir': '../data/TRANCE/event',
    #     'view_final': ['Camera_Center']
    # },
    "standard": {
        "valid_attrs": ["position", "shape", "size", "color", "material"],
        "dir": "../data/TRANCE/standard",
        "view_final": ["Camera_Center"],
    },
    "position": {
        "valid_attrs": ["position"],
        "dir": "../data/TRANCE/position",
        "view_final": ["Camera_Center"],
    },
}


@app.route("/api/default_datadirs/", methods=["GET"])
def get_default_datadirs():
    default_root = Path("../data/TRANCE").resolve()
    res = {"dirs": [], "exists": False}
    if default_root.exists():
        res["dirs"] = sorted(
            [str(x) for x in default_root.glob("*") if x.is_dir()]
        )
        res["exists"] = True
    return res


@app.route("/api/default_configdir/", methods=["GET"])
def get_default_configdir():
    res = {"dir": str(Path("configs").resolve()), "exists": False}
    if Path(res["dir"]).exists():
        res["exists"] = True
    return res


@app.route("/api/infolist/", methods=["GET"])
def get_infolist():
    datadir = request.args.get("dir", "")
    infodir = Path(datadir).expanduser() / "info"
    res = {"infolist": [], "infodir": str(infodir), "exists": False}
    if infodir.exists():
        res["infolist"] = [str(x) for x in list(sorted(infodir.glob("*.json")))]
        res["exists"] = True
    return res


@app.route("/api/info/", methods=["GET"])
def get_info():
    infopath = Path(request.args.get("path", "")).expanduser()
    res = {
        "info": {},
        "infopath": str(infopath),
        "exists": False,
        "error": False,
    }
    try:
        if infopath.exists():
            with open(infopath, "r") as f:
                res["info"] = json.load(f)
            res["exists"] = True
    except Exception:
        res["error"] = True
    return res


@app.route("/api/problems/", methods=["GET"])
def get_problems():
    res = {"problems": list(test_dict.keys())}
    return res


@app.route("/api/random_info/", methods=["GET"])
def get_random_info():
    problem = request.args.get("problem", "basic")
    user = request.args.get("user", "test")
    res = {
        "problem": problem,
        "user": user,
        "path": "",
        "idx": 0,
        "datadir": "",
        "objs": [],
        "pairs": {},
        "pairs_list": [],
        "info": {},
        "view_initial": "Camera_Center",
        "view_final": "",
    }
    if problem in test_dict:
        split = test_dict[problem]
        t = Translator(valid_attrs=split["valid_attrs"])
        res["objs"] = list(range(t.NUMOBJECTS))
        res["pairs"] = t.pairs_dict
        res["pairs_list"] = t.pairs
        res["datadir"] = split["dir"]
        infodir = Path(res["datadir"]) / "info"
        if infodir.exists():
            if str(infodir) not in sample_path:
                sample_path[str(infodir)] = [
                    x.name for x in list(sorted(infodir.glob("*.json")))
                ]

                if problem == "event":
                    sample_path[str(infodir)] = sample_path[str(infodir)][
                        :110000
                    ]

            random.seed(user + problem)
            shuffled_samples = random.sample(
                sample_path[str(infodir)], len(sample_path[str(infodir)])
            )
            unfinished_samples = filter_finished_samples(
                shuffled_samples, split["dir"], user, problem
            )
            infopath = infodir / unfinished_samples[0]
            random.seed(infopath)
            res["view_final"] = random.choice(split["view_final"])
            res["path"] = str(infopath)
            res["idx"] = int(infopath.name.split("_")[-1].split(".")[0])
            if infopath.exists():
                with open(infopath, "r") as f:
                    res["info"] = extract(
                        json.load(f), res["view_initial"], res["view_final"]
                    )
    return res


def filter_finished_samples(samples, datadir, user, problem):
    testdir = Path(datadir) / human_test_dir
    finished_samples = list(testdir.glob("{}.{}.*.json".format(user, problem)))
    finished_samples = [
        x.name[len("{}.{}.".format(user, problem)) :] for x in finished_samples
    ]
    unfinished_samples = list(
        filter(lambda x: x not in finished_samples, samples)
    )
    print(len(sample_path), len(finished_samples), len(unfinished_samples))
    return unfinished_samples


@app.route("/api/submit_test", methods=["POST"])
def submit_test():
    data = request.get_json()
    pred = data["transformations"]
    sample = data["sample"]
    eva = EventEvaluator(
        valid_attrs=test_dict[sample["problem"]]["valid_attrs"]
    )
    with open(sample["path"], "r") as f:
        info = json.load(f)
    sample["info"]["transformations"] = info["transformations"]
    test_dir = Path(sample["datadir"]) / human_test_dir
    if not test_dir.exists():
        test_dir.mkdir(exist_ok=True)
    result_path = test_dir / "{}.{}.{}".format(
        sample["user"], sample["problem"], Path(sample["path"]).name
    )
    res = eva.eval_text_result(info, pred)
    data["result"] = res
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)
    res["user"] = sample["user"]
    res["end_time"] = data["end_time"]
    res["end_timestamp"] = data["end_timestamp"]
    res["duration"] = data["duration"]
    return res


def extract(info, view_initial, view_final):
    brief = {
        "states": [
            {
                "objects": info["states"][0]["objects"],
                "image": info["states"][0]["images"][view_initial],
            },
            {
                "image": info["states"][-1]["images"][view_final],
            },
        ]
    }
    return brief


@app.route("/api/test_history", methods=["GET"])
def get_test_history():
    user = request.args.get("user", "*")
    problem = request.args.get("problem", None)
    res = {"user": user, "samples": {}, "statistics": {}}
    if problem is None:
        problems = list(test_dict.keys())
    else:
        problems = [problem]

    for p in problems:
        test_dir = Path(test_dict[p]["dir"]) / human_test_dir
        history_list = list(test_dir.glob("{}.{}.*.json".format(user, p)))
        samples = []
        for case in history_list:
            with case.open("r") as f:
                samples.append(json.load(f))
        res["samples"][p] = sorted(
            samples, key=lambda x: x["end_timestamp"], reverse=True
        )
        res["statistics"][p] = statistic_result(samples, p)

        if p == "view":
            for view in ["Camera_Left", "Camera_Center", "Camera_Right"]:
                v_samples = [
                    x for x in samples if x["sample"]["view_final"] == view
                ]
                res["statistics"][view] = statistic_result(v_samples, p)

    return res


def statistic_result(samples, problem):
    if len(samples) == 0:
        return {"count": len(samples)}
    metrics = defaultdict(list)
    metrics["step"] = defaultdict(list)
    for s in samples:
        metrics["acc"].append(s["result"]["correct"])
        metrics["loose_acc"].append(s["result"]["loose_correct"])
        metrics["time"].append(s["duration"])
        if problem in ["basic", "view"]:
            t_pred = s["transformations"][0]
            t_gt = s["sample"]["info"]["transformations"][0]
            metrics["obj_acc"].append(t_pred["obj_idx"] == t_gt["obj_idx"])
            metrics["attr_acc"].append(t_pred["attr"] == t_gt["attr"])
            if t_gt["attr"] == "position" and metrics["attr_acc"][-1]:
                target_pred = t_pred["pair"].split(translator.SPLIT)[1:]
                target_pred[1] = int(target_pred[1])
                metrics["val_acc"].append(target_pred in t_gt["options"])
                print(t_pred["pair"].split(translator.SPLIT)[1:])
            else:
                metrics["val_acc"].append(t_pred["pair"] == t_gt["pair"])

        metrics["dist"].append(s["result"]["dist"])
        metrics["norm_dist"].append(s["result"]["norm_dist"])
        length = len(s["sample"]["info"]["transformations"])
        metrics["step"][str(length)].append(s["result"]["correct"])
    res = {
        "count": len(samples),
        "acc": np.array(metrics["acc"]).mean(),
        "loose_acc": np.array(metrics["loose_acc"]).mean(),
        "time": int(np.array(metrics["time"]).mean()),
    }
    if problem in ["basic", "view"]:
        res["obj_acc"] = np.array(metrics["obj_acc"]).mean()
        res["attr_acc"] = np.array(metrics["attr_acc"]).mean()
        res["val_acc"] = np.array(metrics["val_acc"]).mean()

    res["dist"] = np.array(metrics["dist"]).mean()
    res["norm_dist"] = np.array(metrics["norm_dist"]).mean()
    res["step"] = {}
    for step, values in metrics["step"].items():
        res["step"][step] = np.array(metrics["step"][step]).mean()

    return res


@app.route("/api/image/", methods=["GET"])
def get_image():
    imagepath = Path(request.args.get("path", "")).expanduser().resolve()
    if imagepath.exists() and imagepath.suffix == ".png":
        return send_file(imagepath)
    else:
        return "Image not found."


@app.route("/api/configs/", methods=["GET"])
def get_configlist():
    configdir = Path(request.args.get("dir", "")).expanduser()
    res = {
        "configlist": [],
        "configdir": str(configdir),
        "exists": False,
        "configs": [],
    }
    if configdir.exists():
        res["exists"] = True
        yaml = YAML()
        for config in sorted(configdir.glob("*.yaml")):
            if not config.is_symlink():
                res["configlist"].append(str(config))
                info = yaml.load(config)
                info["config"] = str(config)
                res["configs"].append(info)
    return res


@app.route("/api/result/", methods=["GET"])
def get_result():
    configpath = Path(request.args.get("config", "")).expanduser()
    res = {
        "configpath": str(configpath),
        "datatype": "",
        "config_exists": False,
        "result_exists": False,
        "final_views": [],
        "pairs": translator.pairs,
        "attrs": translator.attrs,
    }
    if configpath.exists() and configpath.suffix == ".yaml":
        res["config_exists"] = True
        yaml = YAML()
        config = yaml.load(configpath)
        res["datatype"] = datatype = config["data"]
        resultpath = Path(config["result_path"])
        if resultpath.exists():
            res["result_exists"] = True
            with open(resultpath, "r") as f:
                detail = json.load(f)
            if datatype == "view":
                res["final_views"] = detail.pop("final_views")
            if "event" in datatype:
                res["step_result"] = compute_step_result(detail)
            res["detail"] = [
                dict(zip(["id"] + list(detail.keys()), [i] + list(item)))
                for i, item in enumerate(zip(*detail.values()))
            ]
        return res
    else:
        return "Result not found."


def compute_step_result(result):
    steps = np.unique(np.array(result["target_step"])).tolist()
    step_result = defaultdict(dict)
    for step in steps:
        idx = np.array(result["target_step"]) == step
        step_result[step]["acc"] = np.mean(np.array(result["correct"])[idx])
        step_result[step]["dist"] = np.mean(np.array(result["dist"])[idx])
        step_result[step]["norm_disct"] = np.mean(
            np.array(result["norm_dist"])[idx]
        )
    return step_result


@app.route("/api/test_sample/", methods=["GET"])
def get_test_sample():
    configpath = Path(request.args.get("config", "")).expanduser()
    data_id = int(request.args.get("id", 0))
    res = {
        "id": data_id,
        "configpath": str(configpath),
        "configexists": False,
        "info": "",
    }
    if configpath.exists() and configpath.suffix == ".yaml":
        res["config_exists"] = True
        yaml = YAML()
        config = yaml.load(configpath)
        infopath = config["test_data_args"]["info_path"]
        imagepath = config["test_data_args"]["image_path"]
        if infopath not in info_dict:
            with open(infopath, "rb") as f:
                info_dict[infopath] = pkl.load(f)
        if config["data"] == "view":
            n_view = 3
            data_id = data_id // n_view
        res["info"] = info_dict[infopath]["samples"][data_id]
    return res


@app.route("/api/h5image/", methods=["GET"])
def get_h5image():
    configpath = Path(request.args.get("config", "")).expanduser()
    image = request.args.get("image", "")
    if configpath.exists() and configpath.suffix == ".yaml":
        yaml = YAML()
        config = yaml.load(configpath)
        imagepath = Path(config["test_data_args"]["image_path"])
        if imagepath.exists() and imagepath.suffix == ".h5":
            with h5py.File(imagepath, "r") as f:
                img = f["image"][image][()] if image in f["image"] else None
            if img is not None:
                img = Image.fromarray(img)
                file_object = io.BytesIO()
                img.save(file_object, "PNG")
                file_object.seek(0)
                return send_file(file_object, mimetype="image/PNG")
    return "image not found."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)
