# -*- coding: utf-8 -*-
import json
import math
import random

import numpy as np
from scipy import ndimage


class Option:

    def __init__(self, properties_json, values_json):

        self.load_properties(properties_json)
        self.load_values(values_json)

        self.pm = PositionManger(
            x=(self.coordinate['x_min'], self.coordinate['x_max']),
            y=(self.coordinate['y_min'], self.coordinate['y_max']),
            vis_x=(self.coordinate['vis_x_min'], self.coordinate['vis_x_max']),
            vis_y=(self.coordinate['vis_y_min'], self.coordinate['vis_y_max']),
            min_gap=self.coordinate['min_gap'])

    def load_properties(self, path):
        with open(path, 'r') as f:
            self._json = json.load(f)

        self.shapes = self._json['shape']
        self.color = self._json['color']
        self.rgba = {
            color : [float(c) / 255. for c in rgb] + [1.]
            for color, rgb in self.color.items()}
        self.material = self._json['material']
        self.size = self._json['size']
        self.position = self._json["position"]
        self.seg_colors = self._json["seg_color"]
        self.render_colors = self._json["render_color"]

        self.original_coordinate = self._json['coordinate']
        self.coordinate = {
            k : int(math.ceil(v / self.original_coordinate['unit']))
            for k, v in self.original_coordinate.items()}
        self.move_options = self._gen_move_options()

    def load_values(self, path):
        with open(path, 'r') as f:
            self.values = json.load(f)
        self.SPLIT = '.'
        self.pairs_dict = {}
        for attr, vals in self.values.items():
            if attr == 'position':
                for direction in vals['direction']:
                    for step in vals['step']:
                        self.pairs_dict[
                            self.fmt_act(attr, [direction, step])] = {
                                'attr': attr,
                                'value': [direction, step]
                            }
            else:
                for val in vals:
                    self.pairs_dict[self.fmt_act(attr, val)] = {
                        'attr': attr,
                        'value': [val]
                    }
        self.pairs = sorted(list(self.pairs_dict.keys()))

    def reset(self):
        self.pm.reset()

    def record_obj_addition(self, x, y, radius):
        self.pm.add_obj(x, y, radius)

    def record_obj_remove(self, x, y, radius):
        self.pm.remove_obj(x, y, radius)

    def fmt_act(self, attr, value):
        if attr == 'position':
            direction, step = value
            value = '{}{}{}'.format(direction, self.SPLIT, step)
        return '{}{}{}'.format(attr, self.SPLIT, value)

    def get_attr_value(self, pair):
        attr, value = self.pairs_dict[pair]
        return attr, value

    def _get_val(self, attr, key):
        assert attr in self._json.keys(), 'Unrecognized keys'
        return self._json[attr][key]

    def get_shape_val(self, key):
        return self._get_val('shape', key)

    def get_color_val(self, key):
        return self._get_val('color', key)

    def get_seg_color(self, key):
        return self._get_val('seg_color', key)

    def get_real_size(self, key):
        return self._get_val('size', key)

    def get_size(self, key):
        real_size = self.get_real_size(key)
        val = int(math.ceil(real_size / self.original_coordinate['unit']))
        return val

    def is_bigger(self, key_a, key_b):
        return self.get_real_size(key_a) > self.get_real_size(key_b)

    def get_material_val(self, key):
        return self._get_val('material', key)

    def get_seg_object(self, color):
        for obj, render_color in self.render_colors.items():
            if tuple(color[:3]) == tuple(render_color[:3]):
                return obj
        return None

    def get_move_options(self, x, y):
        visible_position = []
        invisible_position = []
        for option in self.move_options:
            d_x, d_y, direction, step = option
            new_x = x + d_x
            new_y = y + d_y
            if self.is_position_valid(new_x, new_y):
                if self.is_visible(new_x, new_y):
                    visible_position.append((direction, step))
                else:
                    invisible_position.append((direction, step))
        return {'visible': visible_position, 'invisible': invisible_position}

    def _gen_move_options(self):
        """
        the format of option: (delta_x, delta_y, direction, step)
        """
        options = []
        step = self.coordinate['step']
        n_steps = self.position['step']
        for direction, v in self.position['direction'].items():
            options.extend([
                (i * step * v[0], i * step * v[1], direction, i)
                for i in n_steps])
        return options

    def get_delta_position(self, direction, n_step):
        if direction in self.position['direction']:
            v = self.position['direction'][direction]
        else:
            raise NotImplementedError(
                'Unrecognized direction: {}'.format(direction))
        step = self.coordinate['step']
        return [x * n_step * step for x in v]

    def _random_get(self, target, exclude=None):
        assert target in self._json.keys(), 'Unrecognized keys.'
        candidates = self._json[target].copy()
        if exclude is not None:
            candidates.pop(exclude)
        key, value = random.choice(list(sorted(candidates.items())))
        return key, value

    def random_shape(self, exclude=None):
        shape, shape_src = self._random_get('shape', exclude)
        return shape, shape_src

    def random_size(self, exclude=None):
        size, scale = self._random_get('size', exclude)
        return size, scale

    def random_rotation(self):
        theta = math.radians(360. * random.random())
        return theta

    def random_material(self, exclude=None):
        material, material_src = self._random_get('material', exclude)
        return material, material_src

    def random_color(self, exclude=None):
        color, _ = self._random_get('color', exclude)
        rgba = self.rgba[color]
        return color, rgba

    def random_position(self, radius, visible):
        if visible:
            return self.pm.random_visible(radius)
        else:
            return self.pm.random_invisible(radius)

    def pos2real(self, x, y):
        real_x = x * self.original_coordinate['unit']
        real_y = y * self.original_coordinate['unit']
        return real_x, real_y

    def _is_pos_int(self, x, y):
        return ('int' in str(type(x)) and 'int' in str(type(y)))

    def _is_pos_valid(self, x, y):
        return (
            self.coordinate['x_min'] <= x <= self.coordinate['x_max']
            and self.coordinate['y_min'] <= y <= self.coordinate['y_max'])

    def _is_pos_vis(self, x, y):
        return (
            self.coordinate['vis_x_min'] <= x <= self.coordinate['vis_x_max']
            and self.coordinate['vis_y_min'] <= y <= self.coordinate['vis_y_max'])

    def is_position_valid(self, x, y):
        return self._is_pos_int(x, y) and self._is_pos_valid(x, y)

    def is_visible(self, x, y):
        assert self.is_position_valid(x, y), (
            'Position {}, {} is invalid'.format(x, y))
        return self._is_pos_int(x, y) and self._is_pos_vis(x, y)


class PositionManger:

    def __init__(
            self, x=(-10, 10), y=(-10, 10), vis_x=(-5, 5),
            vis_y=(-5, 5), min_gap=1):

        self.x_min, self.x_max = x
        self.y_min, self.y_max = y
        self.vis_x_min, self.vis_x_max = vis_x
        self.vis_y_min, self.vis_y_max = vis_y

        self.idx_x_min, self.idx_x_max = self._pos2idx(*x)
        self.idx_y_min, self.idx_y_max = self._pos2idx(*y)
        self.idx_vis_x_min, self.idx_vis_x_max = self._pos2idx(*vis_x)
        self.idx_vis_y_min, self.idx_vis_y_max = self._pos2idx(*vis_y)

        self.range = (self.idx_x_max + 1, self.idx_y_max + 1)

        self.pos = np.full(self.range, True, dtype=np.uint8)
        self.visible = np.full(self.range, False, dtype=np.uint8)
        self.visible[
            self.idx_vis_x_min:self.idx_vis_x_max+1,
            self.idx_vis_y_min:self.idx_vis_y_max+1] = True

        self._circle_masks = {}
        self.min_gap = min_gap

    def reset(self):
        self.pos = np.full(self.range, True, dtype=np.uint8)

    def get_circular_mask(self, r):
        key = str(r)
        if key not in self._circle_masks:
            h = w = 2 * r + 1
            center = (r, r)

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt(
                (X - center[0]) ** 2 + (Y - center[1]) ** 2)

            mask = dist_from_center <= r
            self._circle_masks[key] = mask
        return self._circle_masks[key]

    def get_mask(self, x, y, r):
        circle_mask = ~self.get_circular_mask(r)
        circle_mask_shape = circle_mask.shape

        idx_x, idx_y = self._pos2idx(x, y)

        top_left = (idx_x - r, idx_y - r)
        bottom_right = (idx_x + r, idx_y + r)

        clip_top_left, bias_top_left = self.clip(*top_left)
        clip_bottom_right, bias_bottom_right = self.clip(*bottom_right)

        self.mask = np.full(self.range, True, dtype=np.uint8)
        self.mask[
            clip_top_left[0]:clip_bottom_right[0]+1,
            clip_top_left[1]:clip_bottom_right[1]+1] = circle_mask[
                bias_top_left[0]:circle_mask_shape[0]+bias_bottom_right[0],
                bias_top_left[1]:circle_mask_shape[1]+bias_bottom_right[1]]

        return self.mask

    def clip(self, x, y):
        new_x = max(self.idx_x_min, min(self.idx_x_max, x))
        new_y = max(self.idx_y_min, min(self.idx_y_max, y))
        bias_x = new_x - x
        bias_y = new_y - y
        return (new_x, new_y), (bias_x, bias_y)

    def _pos2idx(self, x, y):
        return x - self.x_min, y - self.y_min

    def _idx2pos(self, idx_x, idx_y):
        return int(idx_x + self.x_min), int(idx_y + self.y_min)

    def add_obj(self, x, y, radius):
        self.pos = self.pos & self.get_mask(x, y, radius + self.min_gap)

    def remove_obj(self, x, y, radius):
        self.pos = self.pos | (~self.get_mask(x, y, radius + self.min_gap))

    def _random_pos(self, valid):
        candidates = list(np.argwhere(valid))
        if len(candidates) == 0:
            return None
        else:
            idx_x, idx_y = random.choice(candidates)
            return self._idx2pos(idx_x, idx_y)

    def _erosion(self, mat, r):
        if r > 0:
            struct = self.get_circular_mask(r)
            mat = ndimage.binary_erosion(mat, struct, border_value=1)
        return mat

    def random_visible(self, obj_radius=0):
        return self._random_pos(
            self._erosion(self.pos, obj_radius) * self.visible)

    def random_invisible(self, obj_radius=0):
        return self._random_pos(
            self._erosion(self.pos, obj_radius) * (1 - self.visible))

    def random_position(self, obj_radius=0):
        return self._random_pos(self._erosion(self.pos, obj_radius))