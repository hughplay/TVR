import json
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from .components.vectorizer import Vectorizer

CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent


class BaseDataset(Dataset):
    def __init__(
        self,
        data_file: str = "/data/trance/data.h5",
        n_samples: int = None,
        read_raw_image: bool = False,
        image_root: str = "/data/trance/image",
        split: str = "train",
        values_json: str = PROJECT_DIR / "trance/resource/values.json",
        properties_json: str = PROJECT_DIR / "trance/resource/properties.json",
        valid_attrs: List[str] = [
            "position",
            "shape",
            "size",
            "color",
            "material",
        ],
        img_aug: bool = True,
        move_out_aug: bool = True,
    ):

        self.data_file = Path(data_file)
        self.n_samples = n_samples
        self.read_raw_image = read_raw_image
        self.image_root = Path(image_root)
        self.split = split
        self.img_aug = img_aug
        self.move_out_aug = move_out_aug
        assert self.data_file.exists(), f"{self.data_file} is not existed."

        self.vectorizer = Vectorizer(
            values_json=values_json,
            properties_json=properties_json,
            valid_attrs=valid_attrs,
        )

        self.keys = []

    def _read_images(self, initial_img_name, final_img_name):
        initial_state = self.read_image(initial_img_name)
        final_state = self.read_image(final_img_name)
        if self.img_aug:
            initial_state, final_state = self.transform(
                initial_state, final_state
            )
        initial_state = TF.to_tensor(initial_state)
        final_state = TF.to_tensor(final_state)
        return initial_state, final_state

    def _read_trans(self, trans_info):
        obj_idx, pair_idx, options = self.vectorizer.trans2vec(
            trans_info, random_move_out=self.move_out_aug
        )
        return obj_idx, pair_idx, options

    def read_image(self, name):
        if self.read_raw_image:
            return self.read_image_from_file(self.image_root / name)
        else:
            return self.read_image_from_h5(name)

    def read_image_from_h5(self, name):
        with h5py.File(self.data_file, "r") as f:
            img = Image.fromarray(f[self.split]["image"][name][..., :3])
        return img

    def read_image_from_file(self, path, resize_w=160, resize_h=120, channel=3):
        img = Image.open(path).convert("RGB")
        if img.size != (resize_w, resize_h):
            img = img.resize((resize_w, resize_h))
        return img

    def transform(self, *imgs, translation=0.05):
        translate = list(
            translation
            * (2 * np.random.random(2) - 1)
            * np.array([imgs[0].width, imgs[0].height])
        )
        return [
            TF.affine(img, angle=0, translate=translate, scale=1, shear=0)
            for img in imgs
        ]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        raise NotImplementedError


class Basic(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(self.data_file, "r") as f:
            self.keys = eval(f[self.split]["basic_keys"][()])
            if self.n_samples is not None:
                self.keys = self.keys[: self.n_samples]

    def extract_info(self, idx):
        with h5py.File(self.data_file, "r") as f:
            sample_info = json.loads(f[self.split]["data"][self.keys[idx]][()])
        init, fin = sample_info["states"][0], sample_info["states"][-1]

        init_img_name = init["images"]["Camera_Center"]
        fin_img_name = fin["images"]["Camera_Center"]
        init_desc = init["objects"]
        trans = sample_info["transformations"][0]
        return init_img_name, fin_img_name, init_desc, trans

    def __getitem__(self, idx):
        init_img_name, fin_img_name, init_desc, trans = self.extract_info(idx)

        init_img, fin_img = self._read_images(init_img_name, fin_img_name)
        init_mat = self.vectorizer.desc2mat(init_desc)
        obj_target, pair_target, options = self._read_trans(trans)

        sample = {
            "sample_id": idx,
            "init": init_img,
            "fin": fin_img,
            "init_desc": init_mat,
            "obj_target": obj_target,
            "pair_target": pair_target,
            "obj_target_vec": init_mat[obj_target],
            "options": options,
        }
        return sample


class Event(BaseDataset):
    def __init__(self, *args, max_step=4, order_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(self.data_file, "r") as f:
            self.keys = eval(f[self.split]["keys"][()])
            if self.n_samples is not None:
                self.keys = self.keys[: self.n_samples]
        self.final_views = ["Camera_Center", "Camera_Left", "Camera_Right"]
        self.order_aug = order_aug
        self.max_step = max_step

    def extract_info(self, idx, final_view="Camera_Center"):
        with h5py.File(self.data_file, "r") as f:
            sample_info = json.loads(f[self.split]["data"][self.keys[idx]][()])
        init, fin = sample_info["states"][0], sample_info["states"][-1]

        init_img_name = init["images"]["Camera_Center"]
        fin_img_name = fin["images"][final_view]
        view_idx = self.final_views.index(final_view)
        init_desc = init["objects"]
        fin_desc = fin["objects"]
        trans = sample_info["transformations"]
        return (
            init_img_name,
            fin_img_name,
            init_desc,
            fin_desc,
            trans,
            view_idx,
        )

    def info2data(self, info):
        init_img_name, fin_img_name, init_desc, fin_desc, trans, view_idx = info

        init_img, fin_img = self._read_images(init_img_name, fin_img_name)
        init_mat = self.vectorizer.desc2mat(init_desc)
        fin_mat = self.vectorizer.desc2mat(fin_desc)

        (
            obj_target_idx,
            pair_target_idx,
            obj_target_vec,
        ) = self.vectorizer.pack_multitrans(
            trans,
            init_mat,
            self.max_step,
            random_move_out=self.move_out_aug,
            random_order=self.order_aug,
        )

        sample = {
            "view_idx": view_idx,
            "init": init_img,
            "fin": fin_img,
            "init_desc": init_mat,
            "fin_desc": fin_mat,
            "obj_target": obj_target_idx,
            "pair_target": pair_target_idx,
            "obj_target_vec": obj_target_vec,
        }
        return sample

    def __getitem__(self, idx):
        info = self.extract_info(idx)
        sample = self.info2data(info)
        sample["sample_id"] = idx
        return sample


class View(Event):
    def __len__(self):
        return len(self.keys) * len(self.final_views)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.final_views)
        final_view = self.final_views[idx % len(self.final_views)]
        info = self.extract_info(sample_idx, final_view=final_view)
        sample = self.info2data(info)
        sample["sample_id"] = idx
        return sample


class TRANCEDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = False,
        dataset_cfg: Dict[str, Any] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_cfg = dataset_cfg

    def _dataloader(self, split):
        dataset_cfg = self.dataset_cfg[split]
        problem = self.dataset_cfg["problems"][split]
        if problem == "basic":
            dataset = Basic(**dataset_cfg, split=split)
        elif problem == "event":
            dataset = Event(**dataset_cfg, split=split)
        elif problem == "view":
            dataset = View(**dataset_cfg, split=split)
        else:
            raise ValueError(f"Unknown problem {problem}")

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=(split == "train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")
