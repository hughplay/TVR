# -*- coding: utf-8 -*-
import json
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from utils.sample import balanced_sample


class SampleManager:

    def __init__(
            self, valid_init_vis, total_obj, valid_steps, valid_pairs,
            valid_move_type, pair_split='.', vis_x=[-20, 20], vis_y=[-20, 20]):

        self.valid_init_vis = valid_init_vis
        self.valid_steps = valid_steps
        self.valid_objs = list(range(total_obj))
        self.valid_pairs = valid_pairs
        self.valid_move_type = valid_move_type
        self.pair_split = pair_split
        self.vis_x_min, self.vis_x_max = vis_x
        self.vis_y_min, self.vis_y_max = vis_y

        self.top_pair = len(self.valid_pairs) // 2 + 1
        self.top_move_type = len(self.valid_move_type) // 2 + 1

        self.n_init_vis = {k : 0 for k in valid_init_vis}
        self.n_step = {k : 0 for k in valid_steps}
        self.n_obj = {k : 0 for k in self.valid_objs}
        self.n_pair = defaultdict(lambda: defaultdict(int))
        self.n_pair['gram_1'] = {k : 0 for k in valid_pairs}
        self.n_move_type = {k : 0 for k in valid_move_type}

        self._gram_n_pair = None
        self.last_seq_trans = None

        self.n_fail = defaultdict(lambda: defaultdict(int))
        self.c_stage_fail = 0
        self.c_stage_success = 0
        self.c_sample_load = 0
        self.c_sample_fail = 0
        self.c_sample_success = 0

        self.t_start = time.time()
        self.t_sample = None
        self.t_total_success = 0
        self.t_total_fail = 0

        self.EPSILON = 1e-10

        self.reset()

    @property
    def t_total_gen(self):
        return self.t_total_success + self.t_total_fail

    @property
    def time_avg_stage(self):
        return self.t_total_gen / (self.c_stage_success + self.EPSILON)

    @property
    def time_avg_sample(self):
        return self.t_total_gen / (self.c_sample_success + self.EPSILON)

    @property
    def rate_stage_success(self):
        return self.c_stage_success / (
            self.c_stage_success + self.c_stage_fail + self.EPSILON)

    @property
    def rate_sample_success(self):
        return self.c_sample_success / (
            self.c_sample_success + self.c_sample_fail + self.EPSILON)

    @property
    def balance_state(self):
        state = {}
        for gram, stat in self.n_pair.items():
            n_option = len(self.valid_pairs) ** int(gram[-1])
            c_gram = list(stat.values()) + [0] * (
                n_option - len(stat.values()))
            state[gram] = {
                '_options': n_option,
                'mean': float(np.mean(c_gram)),
                'std': float(np.std(c_gram)),
                'min': float(np.min(c_gram)),
                'max': float(np.max(c_gram)),
                'median': float(np.median(c_gram)),
            }
        return state

    @property
    def gram_n_pair(self):
        if self.last_seq_trans is None \
                or self.last_seq_trans != self.seq_trans:
            self._gram_n_pair = self.compute_gram_n_pair()
        return self._gram_n_pair

    def compute_gram_n_pair(self):
        state = self.balance_state
        gram_n_pair = defaultdict(int)
        for pair in self.valid_pairs:
            candidate_pairs = [x[1] for x in self.seq_trans] + [pair]
            for i in range(1, len(candidate_pairs) + 1):
                key = ' '.join(candidate_pairs[-i:])
                gram_n = 'gram_{}'.format(i)
                min_n_pair = state[gram_n]['min'] if gram_n in state else 0
                gram_n_pair[pair] += (self.n_pair[gram_n][key] - min_n_pair)
        return gram_n_pair

    def random_init(self):
        self.init_vis = balanced_sample(self.n_init_vis)
        self.n_init_vis[self.init_vis] += 1
        return self.init_vis

    def random_step(self):
        self.step = balanced_sample(self.n_step)
        self.n_step[self.step] += 1
        return self.step

    def random_obj(self, valid_objs=[]):
        n_obj = {k: v for k, v in self.n_obj.items() if k in valid_objs}
        self.obj = balanced_sample(n_obj)
        return self.obj

    def random_pair(self):
        self.pair = balanced_sample(
            self.gram_n_pair, exclude=self.tried_pairs, top=self.top_pair)
        if self.pair is not None:
            self.tried_pairs.append(self.pair)
        return self.pair

    def random_move_type(self):
        self.move_type = balanced_sample(
            self.n_move_type, exclude=self.tried_move_type,
            top=self.top_move_type)
        if self.move_type is not None:
            self.tried_move_type.append(self.move_type)
        return self.move_type

    def stage_success(self):
        self.n_obj[self.obj] += 1
        if self.pair.startswith('position'):
            self.n_move_type[self.move_type] += 1
            self.seq_trans.append((self.obj, self.pair, self.move_type))
        else:
            self.seq_trans.append((self.obj, self.pair))

        for i in range(1, len(self.seq_trans) + 1):
            pairs = [x[1] for x in self.seq_trans[-i:]]
            self.n_pair['gram_{}'.format(i)][' '.join(pairs)] += 1

        self.reset(stage=True)
        self.tried_pairs = []

    def stage_fail(self):
        self.c_stage_fail += 1
        self.n_fail['pair'][self.pair] += 1

        self.reset(stage=True)

    def sample_success(self):
        self.c_sample_success += 1
        self.c_stage_success += self.step
        self.t_total_success += time.time() - self.t_sample

        self.reset(stage=False)

    def sample_fail(self):
        self.n_init_vis[self.init_vis] -= 1
        self.n_step[self.step] -= 1
        self.record_seq_trans(self.seq_trans, addition=False)
        self.c_sample_fail += 1
        self.t_total_fail += time.time() - self.t_sample

        self.reset(stage=False)

    def reset(self, stage=False):
        self.obj = -1
        self.pair = ''
        self.move_type = ''
        self.tried_move_type = []

        if not stage:
            self.init_vis = 0
            self.step = 0
            self.seq_trans = []
            self.t_sample = time.time()
            self.tried_pairs = []


    def record_seq_trans(self, seq_trans, addition=True):
        diff = 1 if addition else -1
        pairs = []
        for trans in seq_trans:
            if trans[1].startswith('position'):
                obj, pair, move_type = trans
                self.n_move_type[move_type] += diff
            else:
                obj, pair = trans

            self.n_obj[int(obj)] += diff
            pairs.append(pair)
            for i in range(1, len(pairs) + 1):
                self.n_pair['gram_{}'.format(i)][' '.join(pairs[-i:])] += diff

    def record_sample(self, sample):
        seq_trans = []
        for trans in sample['transformations']:
            if trans['attr'] == 'position':
                seq_trans.append(
                    (trans['obj_idx'], trans['pair'], trans['type']))
            else:
                seq_trans.append((trans['obj_idx'], trans['pair']))

        self.n_init_vis[self.count_vis(sample['states'][0]['objects'])] += 1
        self.n_step[len(seq_trans)] += 1
        self.record_seq_trans(seq_trans)
        self.c_sample_load += 1
        return sample

    def record_json(self, path):
        with open(path, 'r') as f:
            sample = json.load(f)
        self.record_sample(sample)
        return sample

    def count_vis(self, objects):
        n_vis = 0
        for obj in objects:
            x, y = obj['position']
            if (self.vis_x_min <= x <= self.vis_x_max
                    and self.vis_y_min <= y <= self.vis_y_max):
                n_vis += 1
        return n_vis

    def state(self):
        return {
            'sample_load': self.c_sample_load,
            'sample_success': self.c_sample_success,
            'stage_success': self.c_stage_success,
            'rate_stage_success': self.rate_stage_success,
            'rate_sample_success': self.rate_sample_success,
            'time': time.time() - self.t_start,
            'time_avg_stage': self.time_avg_stage,
            'time_avg_sample': self.time_avg_sample,
        }


# KEYS_INFO = {
#     'simple': ['idx', 'states', 'transformations', ]
# }

KEYS_OBJECT = {
    'simple': ['color', 'material', 'shape', 'size', 'position'],
    'renderable': ['color', 'material', 'shape', 'size', 'position', 'rotation'],
    'full': ['color', 'material', 'shape', 'size', 'position', 'rotation'],
}

class Recorder:

    def __init__(self, mode='full', ):
        assert mode in ['simple', 'renderable', 'full', 'custom']
        self.mode = mode

    def init(self, scene, idx):
        self.info = {
            'idx': idx,
            'states': [],
            'transformations': [],
            'n_transformation': 0,
            'lamps': {
                key: tuple(lamp.location)
                for key, lamp in scene.lamps.items()},
            'cameras': {
                key: tuple(camera.location)
                for key, camera in scene.cameras.items()},
            'directions': scene.view_directions
        }

    def record_scene(
            self, scene, stage, images_path, segs_path, num_pixels, status):
        stage_info = {
            'stage': stage,
            'objects': deepcopy(scene.objects_description),
            'images': images_path,
            'segs': segs_path,
            'graphs': scene.scene_graph,
            'n_pixels': num_pixels,
            'status': status
        }
        self.info['states'].append(stage_info)

    def record_trans(self, start_stage, end_stage, trans_response):
        trans_info = {
            'start_stage': start_stage,
            'end_stage': end_stage,
        }
        trans_info.update(trans_response)
        self.info['transformations'].append(trans_info)
        self.info['n_transformation'] += 1

    def save(self, path, indent=None):
        try:
            with open(path, 'w') as f:
                json.dump(self.info, f, indent=indent)
            return True
        except:
            return False