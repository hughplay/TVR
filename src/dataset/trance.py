import json
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils import get_logger
from .vectorizer import Vectorizer


class TRANCE(Dataset):

    def __init__(
            self, data_root='../data/trance', split='train',
            values_json='../data/gen_src/resource/values.json',
            properties_json='../data/gen_src/resource/properties.json',
            valid_attrs=['position', 'shape', 'size', 'color', 'material'],
            img_aug=True, move_out_aug=True, default_float_type=np.float32):

        self.data_root = Path(data_root).expanduser()
        self.data_file = self.data_root / 'data.h5'
        self.split = split
        self.img_aug = img_aug
        self.move_out_aug = move_out_aug
        assert self.data_file.is_file(), f'{self.data_file} is not existed.'

        self.vectorizer = Vectorizer(
            values_json=values_json, properties_json=properties_json,
            valid_attrs=valid_attrs, default_float_type=default_float_type)

        self.keys = []

    def prepare_image(self, initial_img_name, final_img_name):
        initial_state = self.read_image(initial_img_name)
        final_state = self.read_image(final_img_name)
        if self.img_aug:
            initial_state, final_state = self.transform(
                initial_state, final_state)
        initial_state, final_state = self.img2arr(initial_state, final_state)
        return initial_state, final_state

    def prepare_trans(self, trans_info):
        obj_idx, pair_idx, options = self.vectorizer.trans2vec(
            trans_info, random_move_out=self.move_out_aug)
        target = (obj_idx, pair_idx)
        return target, options

    def read_image(self, name):
        with h5py.File(self.data_file, 'r') as f:
            img = Image.fromarray(f[self.split]['image'][name][()])
        return img

    def transform(self, *imgs, translation=0.05):
        translate = list(
            translation * (2 * np.random.random(2) - 1) * np.array(
            [imgs[0].width, imgs[0].height]))
        return [
            TF.affine(img, angle=0, translate=translate, scale=1, shear=0)
            for img in imgs]

    def img2arr(self, *imgs):
        return [
            np.array(img)[..., :3].transpose([2, 0, 1]) for img in imgs]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        raise NotImplementedError


class Basic(TRANCE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(self.data_file, 'r') as f:
            self.keys = eval(f[self.split]['basic_keys'][()])

    def extract_info(self, idx):
        with h5py.File(self.data_file, 'r') as f:
            sample_info = json.loads(f[self.split]['data'][self.keys[idx]][()])
        init, fin = sample_info['states'][0], sample_info['states'][-1]

        init_img_name = init['images']['Camera_Center']
        fin_img_name = fin['images']['Camera_Center']
        init_desc = init['objects']
        trans = sample_info['transformations'][0]
        return init_img_name, fin_img_name, init_desc, trans

    def __getitem__(self, idx):
        init_img_name, fin_img_name, init_desc, trans = self.extract_info(idx)

        init_img, fin_img = self.prepare_image(init_img_name, fin_img_name)
        init_mat = self.vectorizer.desc2mat(init_desc)
        target, options = self.prepare_trans(trans)

        sample = {
            'init': init_img,
            'fin':fin_img,
            'init_desc': init_mat,
            'target': target,
            'obj_target_vec': init_mat[target[0]],
            "options": options
        }
        return sample


class Event(TRANCE):
    def __init__(self, *args, max_step=4, order_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(self.data_file, 'r') as f:
            self.keys = eval(f[self.split]['keys'][()])
        self.final_views = ['Camera_Center', 'Camera_Left', 'Camera_Right']
        self.order_aug = order_aug
        self.max_step = max_step

    def extract_info(self, idx, final_view='Camera_Center'):
        with h5py.File(self.data_file, 'r') as f:
            sample_info = json.loads(f[self.split]['data'][self.keys[idx]][()])
        init, fin = sample_info['states'][0], sample_info['states'][-1]

        init_img_name = init['images']['Camera_Center']
        fin_img_name = fin['images'][final_view]
        view_idx = self.final_views.index(final_view)
        init_desc = init['objects']
        fin_desc = fin['objects']
        trans = sample_info['transformations']
        return (
            init_img_name, fin_img_name, init_desc, fin_desc, trans, view_idx)

    def info2data(self, info):
        init_img_name, fin_img_name, init_desc, fin_desc, trans, view_idx \
            = info

        init_img, fin_img = self.prepare_image(init_img_name, fin_img_name)
        init_mat = self.vectorizer.desc2mat(init_desc)
        fin_mat = self.vectorizer.desc2mat(fin_desc)

        obj_target_idx, pair_target_idx, obj_target_vec = \
            self.vectorizer.pack_multitrans(
                trans, init_mat, self.max_step,
                random_move_out=self.move_out_aug, random_order=self.order_aug)

        sample = {
            'view': view_idx,
            'init': init_img,
            'fin':fin_img,
            'init_desc': init_mat,
            'fin_desc': fin_mat,
            'target': (obj_target_idx, pair_target_idx),
            'obj_target_vec': obj_target_vec
        }
        return sample

    def __getitem__(self, idx):
        info = self.extract_info(idx)
        return self.info2data(info)


class View(Event):
    def __len__(self):
        return len(self.keys) * len(self.final_views)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.final_views)
        final_view = self.final_views[idx % len(self.final_views)]
        info = self.extract_info(sample_idx, final_view=final_view)
        return self.info2data(info)
