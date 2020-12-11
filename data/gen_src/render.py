# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import tqdm
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import utils
from utils.record import SampleManager, Recorder
from scene import Scene

"""
This script is used for rendering TRANCE samples.
Render on GPUs:
# blender --background --python render.py -- --config configs/standard.yaml
Render on CPUs:
# blender --background --python render.py -- --config configs/standard.yaml \
    --gpu false --render_tile_size 16

Blender must be installed first. Please refer to `install_blender.sh`.
"""


class Renderer:

    def __init__(self, config):
        self.config = config

        self.prepare_log_and_dir()
        self.prepare_basic_settings()
        self.load_scene()
        self.prepare_sampler()

    def prepare_log_and_dir(self):
        self.log_dir = Path(self.config.output_log_dir).expanduser().joinpath(
            '{}_{}_{}'.format(
                self.config.problem, self.config.start,
                self.config.start + self.config.n_sample - 1))
        self.logger = utils.get_logger(
            __name__, log_dir=self.log_dir, use_tqdm=True)

        self.output_dir = Path(
            self.config.output_data_dir).expanduser().joinpath(
                self.config.problem)
        self.image_dir = self.output_dir.joinpath('image')
        self.blend_dir = self.output_dir.joinpath('blend')
        self.info_dir = self.output_dir.joinpath('info')
        self.seg_dir = self.output_dir.joinpath('seg')

        directories = [self.image_dir, self.seg_dir, self.info_dir]

        if self.config.output_blend:
            directories.append(self.blend_dir)
        if self.config.output_log:
            directories.append(self.log_dir)
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)

    def prepare_basic_settings(self):
        self.recorder = Recorder()

        self.global_seed = utils.hash_seed(
            self.config.problem, self.config.seed)

        self.start = config.start
        self.end = config.start + config.n_sample
        self.current_idx = self.start
        self.current_stage = 0

        self.scene_reload_time = 10

    def load_scene(self):
        self.logger.info('Loading scene into blender...')
        self.scene = Scene(
            self.config.base_scene_file, self.config.shape_dir,
            self.config.material_dir, self.config.properties_json)
        self.scene.set_render_args(
            resolution_x=self.config.width,
            resolution_y=self.config.height,
            tile_x=self.config.render_tile_size,
            tile_y=self.config.render_tile_size,
            gpu=self.config.gpu,
            render_num_samples=self.config.render_num_samples,
            transparent_min_bounces=self.config.transparent_min_bounces,
            transparent_max_bounces=self.config.transparent_max_bounces,
            min_bounces=self.config.min_bounces,
            max_bounces=self.config.max_bounces
        )

        self.valid_cameras = self.scene.cameras.keys()
        self.initial_cameras = list(
            set(self.valid_cameras) & set(self.config.initial_cameras))
        self.final_cameras = list(
            set(self.valid_cameras) & set(self.config.final_cameras))
        self.logger.info('Available cameras for initial state: {}'.format(
            self.initial_cameras))
        self.logger.info('Available cameras for final state: {}'.format(
            self.final_cameras))

    def prepare_sampler(self):
        valid_pairs = [
            pair for pair in self.scene.option.pairs
            if pair.startswith(tuple(self.config.valid_attr))]
        assert config.min_init_vis <= config.max_init_vis <= config.total_obj
        self.sample_manager = SampleManager(
            valid_init_vis=range(config.min_init_vis, config.max_init_vis + 1),
            total_obj=config.total_obj,
            valid_steps=range(config.min_trans, config.max_trans + 1),
            valid_pairs=valid_pairs,
            valid_move_type=['inner', 'out', 'in'],
            pair_split=self.scene.option.SPLIT)

    @property
    def str_stage(self):
        if self.current_stage == 0:
            return 'initial'
        elif self.current_stage == self.num_step:
            return 'final'
        else:
            return '{:02d}'.format(self.current_stage)

    @property
    def images_path(self):
        cameras = self.initial_cameras \
            if self.current_stage == 0 else self.final_cameras
        return {
            cam : os.path.join(
                str(self.image_dir), '{}_{}_img_{:06d}-{}.{}.png'.format(
                    self.config.dataset_prefix, self.config.problem,
                    self.current_idx, self.str_stage, cam))
            for cam in cameras}

    @property
    def segs_path(self):
        return {
            cam : os.path.join(
            str(self.seg_dir), '{}_{}_seg_{:06d}-{}.{}.png'.format(
                self.config.dataset_prefix, self.config.problem,
                self.current_idx, self.str_stage, cam))
            for cam in self.final_cameras}

    @property
    def blend_path(self):
        return os.path.join(
            str(self.blend_dir), '{}_{}_{:06d}-{}.blend'.format(
                self.config.dataset_prefix, self.config.problem,
                self.current_idx, self.str_stage))

    @property
    def json_path(self):
        return os.path.join(str(self.info_dir), '{}_{}_{:06d}.json'.format(
            self.config.dataset_prefix, self.config.problem,
            self.current_idx))

    @property
    def prefix(self):
        return '#{}-{} '.format(self.current_idx, self.current_stage)

    def info(self, message):
        self.logger.info(self.prefix + message)

    def warning(self, message):
        self.logger.warning(self.prefix + message)

    def error(self, message):
        self.logger.error(self.prefix + message, exc_info=True)

    def exclude_output_dir(self, path_dict):
        return {
            key : os.path.basename(path) for key, path in path_dict.items()}

    def set_seed(self):
        seed = [self.global_seed, self.current_idx]
        random.seed(str(seed))
        np.random.seed(seed)

    def run(self, replace_start=-1):
        for self.current_idx in tqdm.trange(self.start, self.end, ncols=80):
            # Reload to avoiding the unknown issue of slowing down.
            if self.current_idx % self.scene_reload_time == 0:
                self.load_scene()

            # Skip existing samples.
            if os.path.isfile(self.json_path):
                info = self.sample_manager.record_json(self.json_path)
                if self.current_idx >= replace_start:
                    self.render_from_json(info)
                continue

            self.logger.info('---')
            self.set_seed()
            while True:
                try:
                    if self.build_sample():
                        self.sample_manager.sample_success()
                        self.summary()
                        break
                    else:
                        self.sample_manager.sample_fail()
                        self.clear_history()
                except KeyboardInterrupt:
                    self.warning('Keyboard Interrupted.')
                    sys.exit(1)
                except Exception as e:
                    self.error(str(e))

    """
    Rule A: no overlapping.
    Rule B: transformation must be observable
    1. occluded objects can not be transformmed (visible in initial states)
    2. transformed objects can not be moved out (visible in final states)
    3. transformed objects can not be occluded (visible in final states)
    """
    def build_sample(self):
        # Sample the initial state.
        self.current_stage = 0
        if not self.sample_init_state():
            return False
        self.record()

        # Decide the number of steps.
        self.num_step = self.sample_manager.random_step()
        self.logger.info(
            'Number of atomic transformations: {}.'.format(self.num_step))

        # Apply randomly sampled atomic transformations one by one.
        for _ in range(self.num_step):
            self.current_stage += 1
            while True:
                pair = self.sample_manager.random_pair()
                if pair is None:
                    return False
                response = self.try_pair(pair)
                if response is None:
                    self.warning(
                        '[Fair] ({}/{}) {} -> no feasible object.'.format(
                            len(self.sample_manager.tried_pairs),
                            self.sample_manager.top_pair, pair))
                    self.sample_manager.stage_fail()
                else:
                    self.record(trans_response=response)
                    self.sample_manager.stage_success()
                    break

        # Show the sequence of transformation.
        for i, trans_info in enumerate(self.recorder.info['transformations']):
            self.logger.info('[{}] {} -> {}{}'.format(
                i, trans_info['pair'], trans_info['obj_idx'],
                ' ({})'.format(trans_info['type'])
                if trans_info['attr'] == 'position' else ''))

        # Save the information about the sample into a json file.
        self.info('Saveing info...')
        if not self.recorder.save(self.json_path, self.config.json_indent):
            return False

        with open(self.json_path, 'r') as f:
            info = json.load(f)
        self.render_from_json(info)

        return True

    def render_from_json(self, info):
        cameras = info['cameras']
        lamps = info['lamps']
        self.num_step = len(info['states']) - 1
        render_stages = list(range(self.num_step + 1))
        if not self.config.render_intermediate:
            render_stages = [render_stages[0], render_stages[-1]]
        for s in render_stages:
            self.current_stage = s
            for _, path in self.images_path.items():
                if not os.path.isfile(path):
                    objects = info['states'][s]['objects']
                    self.scene.set_scene(objects, cameras, lamps)
                    if not self.render_main():
                        return False
                    break
        return True

    def sample_init_state(self):
        self.info('Start to building initial state.')

        # Reset & initialize blender environment.
        self.info('Reset environment.')
        self.scene.reset()
        self.scene.perturb_camera(self.config.camera_jitter)
        self.scene.perturb_lamp(self.config.lamp_jitter)

        # Reset recorder.
        self.info('Reset recorder')
        self.recorder.init(self.scene, self.current_idx)

        # Decide the visibilities of objects.
        self.num_init_visible = self.sample_manager.random_init()
        visible_choices = [True] * self.num_init_visible \
            + [False] * (self.config.total_obj - self.num_init_visible)
        random.shuffle(visible_choices)

        # Randomly create and place objects.
        self.info('Creating {} random objects, {} visible...'.format(
            self.config.max_init_vis, self.num_init_visible))
        for i, visible in enumerate(visible_choices):
            obj = self.scene.create_random_object(visible=visible)
            if obj is None:
                self.warning('{} No enough space for {} objects.'.format(
                    i, 'visible' if visible else 'invisible'))
                self.warning('Failed to build initial state. Retry...')
                return False
            else:
                self.info(
                    '{} ({:^9s}) Put a {:6s} {:6s} {:6s} {:8s}'
                    ' at ({:3d}, {:3d})'.format(
                        i, 'visible' if visible else 'invisible', obj['size'],
                        obj['color'], obj['material'], obj['shape'],
                        *obj['position']))

        self.info('* Successfully build initial state.')
        return True

    def try_pair(self, pair):
        # Try sampled atomic transformations on objects.
        while True:
            move_type = self.sample_manager.random_move_type() \
                if pair.startswith('position') else None
            if pair.startswith('position') and move_type is None:
                return None
            objs = self.get_valid_objs(pair, move_type=move_type)
            self.info('Valid objects for {}: {}'.format(pair, objs))
            while len(objs) > 0:
                obj_idx = self.sample_manager.random_obj(valid_objs=objs)
                objs.remove(obj_idx)
                is_success, response = self.transform(
                    self.scene.objects[obj_idx], pair, move_type)
                if is_success:
                    return response
            if not pair.startswith('position'):
                return None
            else:
                self.warning('[Fail] ({}/{}) move {}'.format(
                    len(self.sample_manager.tried_move_type),
                    self.sample_manager.top_move_type, move_type))

    def get_valid_objs(self, pair, move_type=None):
        # Rule 1. Occluded objects can not be transformmed.
        if move_type == 'in':
            choices = self.scene.invisible_objects
        else:
            choices = self.get_non_occluded_objs()

        choices = [self.scene.get_index(obj) for obj in choices]

        return choices

    def transform(self, obj, pair, move_type=None):
        trans = [obj, *pair.split(self.scene.option.SPLIT)]
        attr = trans.pop(1)

        trans_info = {
            'pair': pair,
            'obj_name': self.scene.get_name(obj),
            'obj_idx': self.scene.get_index(obj)
        }
        trans_func = getattr(self, 't_{}'.format(attr))
        args = trans + [move_type] if move_type else trans
        self.info('{} <- {}'.format(trans_info['obj_name'], pair))
        is_success, response = trans_func(*args)

        response.update(trans_info)

        return is_success, response

    def t_position(self, obj, direction, step, move_type):
        step = int(step)
        response = {
            'attr': 'position',
            'old': self.scene.get_position(obj),
            'new': '',
            'target': (direction, step),
            'type': '',
            'options': [],
        }

        x, y = response['old']
        dx, dy = self.scene.option.get_delta_position(direction, step)
        new_x = x + dx
        new_y = y + dy

        response['new'] = (new_x, new_y)

        if not self.scene.option.is_position_valid(new_x, new_y):
            self.warning('[Fail] position ({}, {}) is invalid.'.format(
                new_x, new_y))
            return False, response

        vis_old = self.scene.option.is_visible(x, y)
        vis_new = self.scene.option.is_visible(new_x, new_y)

        if vis_old and vis_new:
            response['type'] = 'inner'
        elif vis_old and (not vis_new):
            response['type'] = 'out'
        elif (not vis_old) and vis_new:
            response['type'] = 'in'
        else:
            self.warning('[Fail] move object inside invisible area.')
            return False, response

        if response['type'] != move_type:
            self.warning('[Fail] move type ({} vs. {}) not match.'.format(
                response['type'], move_type))
            return False, response

        # Rule 2. Transformed objects can not be moved out.
        if response['type'] == 'out' and self.was_transformed(obj):
            self.warning('[Fail] Modified objects can not be moved out.')
            return False, response

        self.scene.set_position(obj, new_x, new_y)

        if self.scene.is_overlapped(obj):
            self.warning('[Fail] Overlap.')
        # Rule 3. Transformed objects can not be occluded.
        elif self.cause_occlusion(obj):
            self.warning('[Fail] Cause occlusion.')
        else:
            if response['type'] == 'out':
                response['options'].extend(
                    self.scene.option.get_move_options(x, y)['invisible'])
            else:
                response['options'].append((direction, step))
            return True, response

        # Revert.
        self.scene.set_position(obj, x, y)
        return False, response

    def t_shape(self, obj, new_shape):
        response = {
            'attr': 'shape',
            'old': self.scene.get_shape(obj),
            'new': None,
            'target': None,
        }
        old_shape = response['old']
        if new_shape == old_shape:
            self.warning('[Fail] no change.')
            return False, response
        obj = self.scene.set_shape(obj, new_shape)
        response['new'] = response['target'] = new_shape

        return True, response

    def t_size(self, obj, new_size):
        response = {
            'attr': 'size',
            'old': self.scene.get_size(obj),
            'new': new_size,
            'target': new_size,
        }
        old_size = response['old']
        if new_size == old_size:
            self.warning('[Fail] no change.')
            return False, response

        self.scene.set_size(obj, new_size)

        if self.scene.option.is_bigger(new_size, old_size) and \
                self.scene.is_overlapped(obj):
            self.warning('[Fail] Overlap.')
        # Rule 3. transformed objects can not be occluded (visible in final states)
        elif self.cause_occlusion(obj):
            self.warning('[Fail] Cause occlusion.')
        else:
            return True, response

        self.scene.set_size(obj, old_size)
        return False, response

    def t_material(self, obj, new_material):
        response = {
            'attr': 'material',
            'old': self.scene.get_material(obj),
            'new': new_material,
            'target': new_material,
        }
        old_material = response['old']
        if new_material == old_material:
            self.warning('[Fail] no change.')
            return False, response

        self.scene.set_material(obj, new_material)
        return True, response

    def t_color(self, obj, new_color):
        response = {
            'attr': 'color',
            'old': self.scene.get_color(obj),
            'new': new_color,
            'target': new_color,
        }
        old_color = response['old']
        if new_color == old_color:
            self.warning('[Fail] no change.')
            return False, response

        self.scene.set_color(obj, new_color)
        return True, response

    def render_main(self):
        if not self.config.no_render:
            self.info('[Render] main ({})...'.format(self.current_stage))
            for key, image_path in self.images_path.items():
                if not os.path.isfile(image_path):
                    time_used = self.scene.render(
                        image_path, self.scene.cameras[key],
                        config.width, config.height)
                    self.info('- {}: {}'.format(
                        key, utils.time2str(time_used)))
                    if not os.path.isfile(image_path):
                        return False
        return True

    def render_seg(self):
        self.info('[Render] seg ({})...'.format(self.current_stage))
        for key, seg_path in self.segs_path.items():
            time_seg_used = self.scene.render_shadeless(
                seg_path, self.scene.cameras[key],
                config.seg_width, config.seg_height)
            self.info('- {}: {}'.format(
                key, utils.time2str(time_seg_used)))
            if not os.path.isfile(seg_path):
                return False
        return True

    def record(self, trans_response=None):
        status = self.get_objs_status()

        self.recorder.record_scene(
            self.scene, self.current_stage,
            self.exclude_output_dir(self.images_path),
            self.exclude_output_dir(self.segs_path),
            status['n_pixels'], status['status'])
        if trans_response is not None:
            self.recorder.record_trans(
                self.current_stage - 1, self.current_stage, trans_response)

        if self.config.output_blend:
            self.info('[Save] Blender file.')
            self.scene.save(self.blend_path)

    def get_objs_status(self, stage=None):
        if stage is None:
            stage = self.current_stage
        states = self.recorder.info['states']
        assert -len(states) <= stage <= len(states), (
            'stage {} is out of range'.format(stage))
        if -len(states) <= stage < len(states):
            return {
                'status': states[stage]['status'],
                'n_pixels': states[stage]['n_pixels']
            }
        else:
            assert self.render_seg(), '[Error] render seg failed.'
            n_pixels = {}
            cams = list(set(self.initial_cameras) | set(self.final_cameras))
            for cam in cams:
                image_path = self.segs_path[cam]
                n_pixels[cam] = self.count_pixels(image_path)
            status = self.pixels2status(n_pixels)
            return {
                'status': status,
                'n_pixels': n_pixels
            }

    def count_pixels(self, seg_path):
        assert os.path.isfile(seg_path), '[Error] {} doesn\'t exist.'.format(
            seg_path)
        pixels = {}
        colors = utils.count_color(seg_path)
        for n, color in sorted(colors):
            obj = self.scene.option.get_seg_object(color)
            if obj is not None:
                pixels[obj] = n
        return pixels

    def pixels2status(self, n_pixels):
        status = defaultdict(dict)
        min_pixels = self.config.occlusion_threshold
        for cam, val in n_pixels.items():
            for obj in self.scene.objects:
                if obj in self.scene.invisible_objects:
                    status[cam][obj.name] = 'invisible'
                elif obj.name not in val or val[obj.name] < min_pixels:
                    status[cam][obj.name] = 'occluded'
                else:
                    status[cam][obj.name] = 'visible'
        return status

    def cause_occlusion(self, obj):
        status = self.get_objs_status(stage=self.current_stage)['status']
        for x in self.scene.objects:
            if x == obj or self.was_transformed(x):
                x_name = self.scene.get_name(x)
                for cam in self.final_cameras:
                    if status[cam][x_name] == 'occluded':
                        return self.scene.get_index(x)
        return None

    def get_non_occluded_objs(self):
        status = self.get_objs_status(stage=0)['status']
        objects = self.scene.visible_objects.copy()
        for cam in self.initial_cameras:
            for obj_name, state in status[cam].items():
                obj = self.scene.b_objects[obj_name]
                if state == 'occluded' and obj in objects:
                    objects.remove(obj)
        return objects

    def was_transformed(self, obj, attr=None):
        for trans_info in self.recorder.info['transformations']:
            if obj.name == trans_info['obj_name'] and (
                    attr is None or trans_info['attr'] == attr):
                return True
        return False

    def clear_history(self):
        self.logger.info('Removing files after failed...')
        for scene in self.recorder.info['states']:
            for image in scene['images'].values():
                img_path = self.image_dir / image
                if img_path.exists():
                    img_path.unlink()
            for seg in scene['segs'].values():
                seg_path = self.seg_dir / seg
                if seg_path.exists():
                    seg_path.unlink()

    def summary(self):
        self.logger.info('---')
        self.logger.info('Data dir: {}'.format(self.output_dir.resolve()))
        if self.config.output_log:
            self.logger.info('Log dir: {}'.format(self.log_dir.resolve()))

        self.logger.info('---')
        state = self.sample_manager.state()
        self.logger.info('Progress: {} (load: {})'.format(
            state['sample_success'] + state['sample_load'],
            state['sample_load']))
        self.logger.info('Sample success: {} ({:.4f})'.format(
            state['sample_success'], state['rate_sample_success']))
        self.logger.info('Stage success: {} ({:.4f})'.format(
            state['stage_success'], state['rate_stage_success']))
        self.logger.info('Total time: {}'.format(
            utils.time2str(state['time'])))
        self.logger.info('Average sample time: {}'.format(
            utils.time2str(state['time_avg_sample'])))
        self.logger.info('Average stage time: {}'.format(
            utils.time2str(state['time_avg_stage'])))

        self.logger.info('---')
        self.logger.info('Initial Visible Object:')
        for init_vis, num in sorted(self.sample_manager.n_init_vis.items()):
            self.logger.info('- {}: {}'.format(init_vis, num))

        self.logger.info('---')
        self.logger.info('Steps:')
        for key, num in sorted(self.sample_manager.n_step.items()):
            self.logger.info('- {}: {}'.format(key, num))

        self.logger.info('---')
        self.logger.info('Object:')
        for key, num in sorted(self.sample_manager.n_obj.items()):
            self.logger.info('- {}: {}'.format(key, num))

        self.logger.info('---')
        self.logger.info('Pair:')
        for key, num in sorted(self.sample_manager.n_pair['gram_1'].items()):
            self.logger.info('- {}: {}'.format(key, num))

        self.logger.info('---')
        self.logger.info('Move Type:')
        for key, num in sorted(self.sample_manager.n_move_type.items()):
            self.logger.info('- {}: {}'.format(key, num))

        self.logger.info('---')
        self.logger.info('Balance State')
        for key, state in sorted(self.sample_manager.balance_state.items()):
            self.logger.info('# {}'.format(key))
            for k, v in sorted(state.items()):
                self.logger.info('- {}: {}'.format(k, v))


if __name__ == '__main__':
    config = utils.Config('configs/standard.yaml')

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--info', action='help', default=argparse.SUPPRESS,
        help='show this help message and exit.')
    parser.add_argument('-c', '--config', help='YAML config file.')
    parser.add_argument(
        '--replace_start', default=-1, type=int,
        help='Replace possibly existed samples from <replace_start>.')
    config.extend_parser(parser)

    argv = []
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        argv = sys.argv[(idx + 1):]
    args = parser.parse_args(argv)
    if args.config:
        config = utils.Config(args.config)
    config.merge_args(args)

    render = Renderer(config)
    render.run(args.replace_start)