import json
import random
from collections import defaultdict
from itertools import accumulate

import numpy as np


class Vectorizer:

    def __init__(
            self, values_json='../data/gen_src/resource/values.json',
            properties_json='../data/gen_src/resource/properties.json',
            default_float_type=np.float32, default_int_type=np.int64,
            split='.',
            valid_attrs=['position', 'shape', 'size', 'color', 'material']):

        self.values_json = values_json
        self.properties_json = properties_json
        self.default_float_type = default_float_type
        self.SPLIT = split
        self.valid_attrs = valid_attrs

        self._prepare_env_settings()
        self._prepare_obj()
        self._prepare_pairs()
        self._define_feature_order()

    def _prepare_env_settings(self):
        with open(self.properties_json, 'r') as f:
            self.properties = json.load(f)
        unit = self.properties['coordinate']['unit']
        self.coord = {
            k : v / unit for k, v in self.properties['coordinate'].items()}
        self.pos_min = self.coord['x_min']
        self.pos_max = self.coord['x_max']

        for s, val in self.properties['size'].items():
            self.coord[s] = val / unit

    def _prepare_obj(self):
        self.NUMOBJECTS = 10
        self.OBJ_EOS = self.NUMOBJECTS
        self.OBJ_PAD = self.NUMOBJECTS + 1
        self.OBJ_SOS = self.NUMOBJECTS + 2

    def _prepare_pairs(self):
        with open(self.values_json, 'r') as f:
            self.values = json.load(f)
        self.shape = self.values['shape']
        self.size = self.values['size']
        self.material = self.values['material']
        self.color = self.values['color']

        self.n_shapes = len(self.shape)
        self.n_sizes = len(self.size)
        self.n_materials = len(self.material)
        self.n_colors = len(self.color)

        self.pairs = []
        self.pairs_dict = defaultdict(list)
        self.attrs = []
        for attr, values in sorted(self.values.items()):
            if attr in self.valid_attrs:
                self.attrs.append(attr)
                if attr == 'position':
                    for direction in values['direction']:
                        for step in values['step']:
                            pair = self.fmt_pair(attr, (direction, step))
                            self.pairs.append(pair)
                            self.pairs_dict[attr].append(pair)
                else:
                    for val in values:
                        pair = self.fmt_pair(attr, val)
                        self.pairs.append(pair)
                        self.pairs_dict[attr].append(pair)
        self.n_attr = len(self.attrs)
        self.n_pairs = len(self.pairs)
        self.PAIR_EOS = self.n_pairs
        self.PAIR_PAD = self.n_pairs + 1
        self.PAIR_SOS = self.n_pairs + 2

    def _define_feature_order(self):
        self.dim_position = 2
        self.feat_order = [
            'position', 'material', 'color', 'shape', 'size']
        self.dim_obj = [
            self.dim_position, self.n_materials, self.n_colors,
            self.n_shapes, self.n_sizes]
        self.feat_pos = [0] + list(accumulate(self.dim_obj))
        self.feat_start = {
            k : v for k, v in zip(self.feat_order, self.feat_pos)}
        self.feat_end = {
            k : v for k, v in zip(self.feat_order, self.feat_pos[1:])}
        self.c_obj = sum(self.dim_obj)

    def desc2mat(self, desc):
        return np.array(
            [self.obj2vec(obj) for obj in desc], dtype=self.default_float_type)

    def obj2vec(self, obj):
        vectors = {
            'shape': self._create(
                self.n_shapes, self.shape.index(obj['shape'])),
            'size': self._create(self.n_sizes, self.size.index(obj['size'])),
            'material': self._create(
                self.n_materials, self.material.index(obj['material'])),
            'color': self._create(
                self.n_colors, self.color.index(obj['color'])),
            'position': self.norm_position(
                np.array(obj['position'], dtype=self.default_float_type))
        }
        cluster = [vectors[feat] for feat in self.feat_order]
        feat = np.concatenate(cluster)
        return feat

    def trans2vec(self, trans_info, random_move_out=False):
        obj_idx = trans_info['obj_idx']
        pair = self.fmt_pair(trans_info['attr'], trans_info['val'])
        pair_idx = self.pairs.index(pair)

        options = self.get_options(trans_info)
        option_vec = np.zeros(self.n_pairs, dtype=np.bool)
        for option in options:
            option_vec[option] = 1

        return obj_idx, pair_idx, option_vec
    
    def vec2trans(self, obj_idx, pair_idx):
        pair = self.pairs[pair_idx]
        attr, val = self.sep_pair(pair)
        return {
            'obj': obj_idx,
            'pair': self.pairs[pair_idx],
            'attr': attr,
            'val': val
        }

    def get_options(self, trans_info):
        if 'options' in trans_info:
            attr = trans_info['attr']
            options = [
                self.pairs.index(self.fmt_pair(attr, option))
                for option in trans_info['options']]
        else:
            pair = self.fmt_pair(trans_info['attr'], trans_info['val'])
            options = [self.pairs.index(pair)]
        return options

    def pack_multitrans(
            self, trans, init_mat, max_step,
            random_move_out=False, random_order=False):
        target_obj_idx = np.full(
            (max_step + 1,), self.OBJ_PAD, dtype=np.int64)
        target_obj_vec = np.zeros(
            (max_step + 1, self.c_obj), dtype=self.default_float_type)
        target_pair_idx = np.full(
            (max_step + 1,), self.PAIR_PAD, dtype=np.int64)

        if random_order:
            trans = trans.copy()
            random.shuffle(trans)
        for i, t in enumerate(trans):
            obj_idx, pair_idx, _ = self.trans2vec(
                t, random_move_out=random_move_out)
            target_obj_idx[i] = obj_idx
            target_obj_vec[i] = init_mat[obj_idx]
            target_pair_idx[i] = pair_idx
        target_obj_idx[i + 1] = self.OBJ_EOS
        target_pair_idx[i + 1] = self.PAIR_EOS

        return target_obj_idx, target_pair_idx, target_obj_vec

    def multitrans2vec(self, trans):
        return [self.trans2vec(t)[:2] for t in trans]

    def vec2obj(self, vector):
        pos_split = self.feat_pos[1:-1]
        splits = np.split(vector, pos_split)

        attr_splits = {
            key: splits[i] for i, key in enumerate(self.feat_order)
        }

        obj = {}
        for attr, split in attr_splits.items():
            if attr == 'position':
                obj[attr] = tuple(self.restore_position(split))
            else:
                obj[attr] = getattr(self, attr)[np.argmax(split)]

        return obj

    def mat2objs(self, mat):
        return [self.vec2obj(vec) for vec in mat]
    
    def vec2multitrans(self, vectors):
        return [self.vec2trans(*v) for v in vectors]

    def _create(self, n_dim, val_dim=None):
        t = np.zeros(n_dim, dtype=np.bool)
        if val_dim is not None:
            t[val_dim] = 1
        return t

    def norm_position(self, pos):
        return (pos - self.pos_min) / (self.pos_max - self.pos_min)

    def restore_position(self, pos):
        res = (pos * (self.pos_max - self.pos_min) + self.pos_min)
        return res.astype(int) if type(res) is np.ndarray else res

    def fmt_pair(self, attr, value):
        if attr == 'position':
            return self.SPLIT.join([attr, value[0], str(value[1])])
        else:
            return self.SPLIT.join([attr, value])

    def sep_pair(self, pair):
        if type(pair) is int:
            pair = self.pairs[pair]
        splits = pair.split(self.SPLIT)
        attr = splits[0]
        if attr == 'position':
            value = (splits[1], int(splits[2]))
        else:
            value = splits[1]
        return attr, value