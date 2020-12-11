# -*- coding: utf-8 -*-
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np

from utils.option import Option

import bpy
import bpy_extras
from mathutils import Vector

"""
This script defines the basic environment for constructing TRANCE.
"""


class Scene:

    def __init__(
            self,
            blend_file='resource/base_scene.blend',
            shape_dir='resource/shapes',
            material_dir='resource/materials',
            properties_json='resource/properties.json',
            values_json='resource/values.json'):

        self.blend_file = blend_file
        self.shape_dir = shape_dir
        self.material_dir = material_dir
        self.properties_json = properties_json
        self.values_json = values_json

        self.objects = []
        self.objects_description = []
        self.visible_objects = []
        self.invisible_objects = []

        self.cameras = {}
        self.cameras_init_loc = {}
        self.view_directions = None
        self.lamps = {}
        self.lamps_init_loc = {}

        self.creating = False
        self.option = Option(self.properties_json, self.values_json)

        self.load_resource()
        self.set_render_args()
        self.record_init_location()

        self.DEFAULT_CAMERA = 'Camera_Center'
        self.switch_camera(self.cameras[self.DEFAULT_CAMERA])

    @property
    def b_objects(self):
        return bpy.data.objects

    @property
    def materials(self):
        return bpy.data.materials

    @property
    def current_camera(self):
        return bpy.context.scene.camera

    @current_camera.setter
    def current_camera(self, camera):
        bpy.context.scene.camera = camera

    def reset(self):
        for obj in self.objects:
            self.b_objects.remove(obj)
        for material in self.materials:
            self.materials.remove(material)

        self.visible_objects.clear()
        self.invisible_objects.clear()
        self.objects.clear()
        self.objects_description.clear()
        self.option.reset()

        for key, camera in self.cameras.items():
                camera.location = Vector(self.cameras_init_loc[key])

        for key, lamp in self.lamps.items():
                lamp.location = Vector(self.lamps_init_loc[key])

        self.switch_camera(self.cameras[self.DEFAULT_CAMERA])
        self.view_directions = self._compute_directions()

        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

    def load_resource(self):
        # Load the scene from file.
        bpy.ops.wm.open_mainfile(filepath=self.blend_file)

        # Load materials
        for f in os.listdir(self.material_dir):
            if f.endswith('.blend'):
                name = os.path.splitext(f)[0]
                path = os.path.join(self.material_dir, f, 'NodeTree', name)
                bpy.ops.wm.append(filename=path)

    def set_render_args(
            self, engine='CYCLES', resolution_x=320, resolution_y=240,
            resolution_percentage=100, tile_x=256, tile_y=256,
            sample_as_light=True, blur_glossy=2., render_num_samples=512,
            transparent_min_bounces=8, transparent_max_bounces=8,
            min_bounces=0, max_bounces=4, gpu=False):
        render_args = bpy.context.scene.render
        render_args.engine = 'CYCLES'
        render_args.resolution_x = resolution_x
        render_args.resolution_y = resolution_y
        render_args.resolution_percentage = resolution_percentage
        render_args.tile_x = tile_x
        render_args.tile_y = tile_y

        bpy.data.worlds['World'].cycles.sample_as_light = sample_as_light
        cycles_args = bpy.context.scene.cycles
        cycles_args.blur_glossy = blur_glossy
        cycles_args.samples = render_num_samples
        cycles_args.transparent_min_bounces = transparent_min_bounces
        cycles_args.transparent_max_bounces = transparent_max_bounces
        cycles_args.min_bounces = min_bounces
        cycles_args.max_bounces = max_bounces

        prefs = bpy.context.user_preferences.addons['cycles'].preferences
        if gpu == 'auto':
            gpu = len(self.gpus()) > 0
        if gpu is True:
            prefs.compute_device_type = 'CUDA'
            cycles_args.device = 'GPU'
        else:
            prefs.compute_device_type = 'NONE'
            cycles_args.device = 'CPU'

    def gpus(self):
        prefs = bpy.context.user_preferences.addons['cycles'].preferences
        available_gpus = []
        for device in prefs.devices:
            if device.type == 'CUDA':
                available_gpus.append(device)
        return available_gpus

    def record_init_location(self):
        for key in ['Camera_Left', 'Camera_Center', 'Camera_Right']:
            self.cameras[key] = self.b_objects[key]
            self.cameras_init_loc[key] = self.cameras[key].location.copy()
        for key in ['Lamp_Key', 'Lamp_Back', 'Lamp_Fill']:
            self.lamps[key] = self.b_objects[key]
            self.lamps_init_loc[key] = self.lamps[key].location.copy()

    def select(self, obj):
        bpy.context.scene.objects.active = obj

    def switch_camera(self, camera):
        assert camera in self.cameras.values()
        self.current_camera = camera

    def set_scene(self, objects, cameras, lamps):
        self.reset()
        for key, loc in cameras.items():
            self.cameras[key].location = Vector(loc)
        for key, loc in lamps.items():
            self.lamps[key].location = Vector(loc)
        for obj in objects:
            self.add_object(
                obj['shape'], obj['size'], obj['rotation'], obj['position'][0],
                obj['position'][1], obj['material'], obj['color'], obj['name'])

    def perturb_camera(self, jitter=0.5):
        for key, camera in sorted(self.cameras.items()):
            for i in range(3):
                camera.location[i] = self.cameras_init_loc[key][i] + \
                    (np.random.random() - 0.5) * 2. * jitter
        self.view_directions = self._compute_directions()

    def perturb_lamp(self, jitter=1.):
        for key, lamp in sorted(self.lamps.items()):
            for i in range(3):
                lamp.location[i] = self.lamps_init_loc[key][i] + \
                    (random.random() - 0.5) * 2. * jitter

    def create_random_object(self, visible=True):
        shape, _ = self.option.random_shape()
        size, _ = self.option.random_size()
        theta = self.option.random_rotation()
        radius = self.option.get_size(size)
        position = self.option.random_position(radius, visible)
        if position is None:
            return None
        material, _ = self.option.random_material()
        color, _ = self.option.random_color()

        obj, obj_description = self.add_object(
            shape, size, theta, *position, material, color)

        return obj_description

    def add_object(
            self, shape, size, theta, x, y, material, color,
            name=None, record=True):

        self.creating = True

        obj = self.add_shape(shape, name)
        self.set_size(obj, size, shape=shape)
        self.set_rotation(obj, theta)
        self.set_position(obj, x, y, record=record)
        self.set_material(obj, material)
        self.set_color(obj, color)

        obj_description = {
            'name': obj.name,
            'shape': shape,
            'size': size,
            'rotation': theta,
            'position': (x, y),
            '3d_coords': tuple(obj.location),
            'view_coords': self.get_view_coords(obj),
            'material': material,
            'color': color
        }

        if record:
            self.objects.append(obj)
            self.objects_description.append(obj_description)
            radius = self.option.get_size(size)
            self.option.record_obj_addition(x, y, radius)

        self.creating = False

        return obj, obj_description

    def remove_object(self, obj):
        assert obj in self.objects, '{} is not in objects.'.format(str(obj))

        idx = self.objects.index(obj)
        x, y = self.get_position(obj)
        radius = self.get_radius(obj)
        self.option.record_obj_remove(x, y, radius)

        self.objects.remove(obj)
        self.objects_description.pop(idx)

        if obj in self.invisible_objects:
            self.invisible_objects.remove(obj)
        if obj in self.visible_objects:
            self.visible_objects.remove(obj)

        self.b_objects.remove(obj, do_unlink=True)

    def replace_object(self, old_obj, new_obj, new_obj_desc):
        assert old_obj in self.objects, (
            '{} is not in objects.'.format(str(old_obj)))

        x, y = self.get_position(old_obj)
        radius = self.get_radius(old_obj)
        new_x, new_y = new_obj_desc['position']
        new_radius = self.option.get_size(new_obj_desc['size'])
        self.option.record_obj_remove(x, y, radius)
        self.option.record_obj_addition(new_x, new_y, new_radius)

        idx = self.objects.index(old_obj)
        self.objects[idx] = new_obj
        self.objects_description[idx] = new_obj_desc

        if old_obj in self.visible_objects:
            self.visible_objects[
                self.visible_objects.index(old_obj)] = new_obj
        if old_obj in self.invisible_objects:
            self.invisible_objects[
                self.invisible_objects.index(old_obj)] = new_obj

        self.b_objects.remove(old_obj, do_unlink=True)

    def add_shape(self, shape, name=None):
        shape_src = self.option.shapes[shape]
        if name is None:
            obj_new_name = 'Object_{}'.format(len(self.objects))
        else:
            obj_new_name = name

        filename = os.path.join(
            self.shape_dir, '{}.blend'.format(shape_src), 'Object', shape_src)
        bpy.ops.wm.append(filename=filename)
        obj = self.b_objects[shape_src]
        obj.name = obj_new_name

        return obj

    def set_shape(self, obj, shape):
        # Use a new object to replace the old object.
        old_name = obj.name
        obj.name = 'Removing'
        idx = self.objects.index(obj)
        obj_desc = self.objects_description[idx]
        new_obj, new_obj_desc = self.add_object(
            shape, obj_desc['size'], obj_desc['rotation'],
            obj_desc['position'][0], obj_desc['position'][1],
            obj_desc['material'], obj_desc['color'],
            name=old_name, record=False)

        for key in obj_desc:
            if key == '3d_coords':
                assert obj_desc[key][:2] == new_obj_desc[key][:2], (
                    'position is different: {} vs {}'.format(
                        obj_desc[key][:2], new_obj_desc[key][:2]))
            elif key not in ['shape', 'view_coords']:
                assert obj_desc[key] == new_obj_desc[key], (
                    '{} is different: {} vs {}'.format(
                        key, obj_desc[key], new_obj_desc[key]))

        self.replace_object(obj, new_obj, new_obj_desc)

        return new_obj

    def set_position(self, obj, x, y, record=True):
        assert self.option.is_position_valid(x, y), (
            'invalid position: ({}, {})'.format(x, y))

        if not self.creating:
            old_x, old_y = self.get_position(obj)
            radius = self.get_radius(obj)

        real_x, real_y = self.option.pos2real(x, y)
        z = obj.scale[2]
        obj.location = Vector((real_x, real_y, z))
        obj.hide_render = not self.option.is_visible(x, y)

        if record:
            if self.option.is_visible(x, y):
                if obj not in self.visible_objects:
                    self.visible_objects.append(obj)
                if obj in self.invisible_objects:
                    self.invisible_objects.remove(obj)
            else:
                if obj not in self.invisible_objects:
                    self.invisible_objects.append(obj)
                if obj in self.visible_objects:
                    self.visible_objects.remove(obj)

        if not self.creating:
            self.modify_desc(obj, '3d_coords', tuple(obj.location))
            self.modify_desc(obj, 'position', (x, y))
            self.modify_desc(
                obj, 'view_coords', self.get_view_coords(obj))
            self.option.record_obj_remove(old_x, old_y, radius)
            self.option.record_obj_addition(x, y, radius)

        return obj

    def set_rotation(self, obj, theta):
        obj.rotation_euler[2] = theta
        if not self.creating:
            self.modify_desc(obj, 'rotation', theta)

        return obj

    def set_size(self, obj, size, shape=None):
        if self.creating:
            assert shape is not None, (
                'Must tell the shape of object when creating')
        else:
            shape = self.get_shape(obj)
            x, y = self.get_position(obj)
            old_radius = self.get_radius(obj)

        scale = self.option.get_real_size(size)
        # Cubes are scaled so that the maximum radius are equal between objects
        # with the same size.
        if shape == 'cube':
            scale /= math.sqrt(2)

        obj.scale = Vector(tuple([scale] * 3))
        obj.location[2] = scale

        if not self.creating:
            self.modify_desc(obj, 'size', size)
            self.modify_desc(obj, '3d_coords', tuple(obj.location))

            radius = self.option.get_size(size)
            self.option.record_obj_remove(x, y, old_radius)
            self.option.record_obj_addition(x, y, radius)

        return obj

    def set_material(self, obj, material):
        material_src = self.option.material[material]

        if not self.creating:
            old_material = obj.data.materials[0]
            material_name = old_material.name
            self.select(obj)
            bpy.ops.object.material_slot_remove()
            self.materials.remove(old_material)

            color = self.describe(obj)['color']
        else:
            material_name = 'Material_{}'.format(len(self.materials))

        bpy.ops.material.new()
        b_material = self.materials['Material']
        b_material.name = material_name
        obj.data.materials.append(b_material)

        group_node = b_material.node_tree.nodes.new('ShaderNodeGroup')
        group_node.node_tree = bpy.data.node_groups[material_src]

        output_node = b_material.node_tree.nodes['Material Output']

        b_material.node_tree.links.new(
            group_node.outputs['Shader'], output_node.inputs['Surface'])

        if not self.creating:
            self.set_color(obj, color)
            self.modify_desc(obj, 'material', material)

        return obj

    def set_color(self, obj, color):
        rgba = self.option.rgba[color]
        inputs = obj.data.materials[0].node_tree.nodes['Group'].inputs
        changed = False
        for inp in inputs:
            if inp.name == 'Color':
                inp.default_value = rgba
                if not self.creating:
                    self.modify_desc(obj, 'color', color)
                changed = True
        assert changed is True

        return obj

    def set_attr(self, attr, *args):
        if attr == 'position':
            self.set_position(*args)
        elif attr == 'shape':
            self.set_shape(*args)
        elif attr == 'size':
            self.set_size(*args)
        elif attr == 'material':
            self.set_material(*args)
        elif attr == 'color':
            self.set_color(*args)
        elif attr == 'rotation':
            self.set_rotation(*args)
        else:
            raise NotImplementedError('Unrecognized attr: {}'.format(attr))

    def is_overlapped(self, obj):
        # Whether obj is overlapped with other objects. 
        x, y = self.get_real_position(obj)
        r = self.option.get_real_size(self.get_size(obj))
        for other_obj in self.objects:
            if other_obj is obj:
                continue
            x_, y_ = self.get_real_position(other_obj)
            r_ = self.option.get_real_size(
                self.get_size(other_obj))

            center_distance = math.sqrt((x - x_) ** 2 + (y - y_) ** 2)
            min_distance = r + r_ + self.option.original_coordinate['min_gap']
            if center_distance < min_distance:
                return True
        return False

    def modify_desc(self, obj, key, value):
        idx = self.objects.index(obj)
        self.objects_description[idx][key] = value
        return self.objects_description[idx]

    def render(self, image_path, camera=None, width=320, height=240):
        render_args = bpy.context.scene.render
        render_args.resolution_x = width
        render_args.resolution_y = height
        if camera is not None:
            assert camera in self.cameras.values(), (
                'Unknown camera: {}'.format(camera))
            old_camera = self.current_camera
            self.switch_camera(camera)

        start = time.time()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)

        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)

        os.close(1)
        os.dup(old)
        os.close(old)
        end = time.time()

        if camera is not None:
            self.switch_camera(old_camera)

        return end - start

    def render_shadeless(self, image_path, camera=None, width=320, height=240):
        # Render segmentation map.
        render_args = bpy.context.scene.render
        render_args.resolution_x = width
        render_args.resolution_y = height
        render_args = bpy.context.scene.render

        old_engine = render_args.engine
        old_use_antialiasing = render_args.use_antialiasing
        old_use_raytrace = render_args.use_raytrace

        render_args.engine = 'BLENDER_RENDER'
        render_args.use_antialiasing = False
        render_args.use_raytrace = False

        hide_objs = list(self.lamps.values()) + [self.b_objects['Ground']]
        for obj in hide_objs:
            obj.hide_render = True

        old_materials = []
        new_materials = []
        for i, obj in enumerate(self.visible_objects):
            old_materials.append(obj.data.materials[0])
            bpy.ops.material.new()
            material = bpy.data.materials['Material']
            new_materials.append(material)
            material.name = 'Material_Tmp_{}'.format(i)
            r, g, b = self.option.get_seg_color(self.get_name(obj))
            material.diffuse_color = [r / 255., g / 255., b / 255.]
            material.use_shadeless = True
            obj.data.materials[0] = material

        render_time = self.render(image_path, camera)

        for obj, material, new_material in zip(
                self.visible_objects, old_materials, new_materials):
            obj.data.materials[0] = material
            self.materials.remove(new_material)

        for obj in hide_objs:
            obj.hide_render = False

        render_args.engine = old_engine
        render_args.use_antialiasing = old_use_antialiasing
        render_args.use_raytrace = old_use_raytrace

        return render_time

    def save(self, blend_path):
        if os.path.isfile(blend_path):
            os.remove(blend_path)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    def _compute_directions(self):
        # Put a plane on the ground so we can compute cardinal directions
        bpy.ops.mesh.primitive_plane_add(radius=5)
        plane = bpy.context.object

        directions = {}

        for name, camera in self.cameras.items():
            # Directions depend on the view angle of the cameras.
            plane_normal = plane.data.vertices[0].normal
            quaternion = camera.matrix_world.to_quaternion()

            cam_behind = quaternion * Vector((0, 0, -1))
            cam_left = quaternion * Vector((-1, 0, 0))

            behind = (
                cam_behind - cam_behind.project(plane_normal)).normalized()
            left = (cam_left - cam_left.project(plane_normal)).normalized()
            behind_left = (behind + left).normalized()
            behind_right = (behind - left).normalized()

            # Return six axis-aligned directions
            directions[name] = {
                'behind': tuple(behind),
                'front': tuple(-behind),
                'left': tuple(left),
                'right': tuple(-left),
                'behind_left': tuple(behind_left),
                'behind_right': tuple(behind_right),
                'front_left': tuple(-behind_right),
                'front_right': tuple(-behind_left)
            }

        self.b_objects.remove(plane, do_unlink=True)

        return directions

    @property
    def scene_graph(self, eps=0.2):
        view_relation = {}
        for camera_name, directions in self.view_directions.items():
            relationships = defaultdict(dict)
            for name in ['left', 'right', 'front', 'behind']:
                vector = directions[name]
                for current_obj in self.visible_objects:
                    current_idx = self.get_index(current_obj)
                    related = set()
                    current_position = self.get_3d_coords(current_obj)
                    for obj in self.visible_objects:
                        if current_obj == obj:
                            continue
                        related_idx = self.get_index(obj)
                        related_position = self.get_3d_coords(obj)
                        diff = [
                            related_position[k] - current_position[k]
                            for k in [0, 1, 2]]
                        dot = sum(diff[k] * vector[k] for k in [0, 1, 2])
                        if dot > eps:
                            related.add(related_idx)
                    relationships[name][str(current_idx)] = sorted(list(related))
            view_relation[camera_name] = relationships
        return view_relation

    def get_view_coords(self, obj):
        view_coords = {}
        for name, camera in self.cameras.items():
            scene = bpy.context.scene
            x, y, z = bpy_extras.object_utils.world_to_camera_view(
                scene, camera, obj.location)
            scale = scene.render.resolution_percentage / 100.
            width = int(scale * scene.render.resolution_x)
            height = int(scale * scene.render.resolution_y)
            view_x =int(round(x * width))
            view_y =int(round(y * height))
            view_coords[name] = (view_x, view_y, z)
        return view_coords

    def get_index(self, obj):
        if obj in self.objects:
            return self.objects.index(obj)
        return None

    def describe(self, obj):
        idx = self.get_index(obj)
        if idx is not None:
            return self.objects_description[idx]
        return None

    def describe_by_index(self, idx):
        assert idx in range(len(self.objects)), (
            'index ({}) exceed range (0-{})'.format(
                idx, len(self.objects) - 1))
        return self.objects_description[idx]

    def get_object_by_name(self, name):
        return self.b_objects[name]

    def get(self, obj, key):
        description = self.describe(obj)
        if description is None:
            return None
        else:
            if key == 'name':
                return description['name']
            elif key == 'shape':
                return description['shape']
            elif key == 'size':
                return description['size']
            elif key == 'rotation':
                return description['rotation']
            elif key == '3d_coords':
                return description['3d_coords']
            elif key == 'real_position':
                return description['3d_coords'][:2]
            elif key == 'position':
                return description['position']
            elif key == 'material':
                return description['material']
            elif key == 'color':
                return description['color']
            else:
                raise NotImplementedError('Unrecognized key: {}'.format(key))

    def get_name(self, obj):
        if obj in self.objects:
            name = self.get(obj, 'name')
            assert name == obj.name, (
                'Name is conflict: {} vs. {}'.format(name, obj.name))
        return obj.name

    def get_shape(self, obj):
        return self.get(obj, 'shape')

    def get_size(self, obj):
        return self.get(obj, 'size')

    def get_rotation(self, obj):
        return self.get(obj, 'rotation')

    def get_real_position(self, obj):
        return self.get(obj, 'real_position')

    def get_position(self, obj):
        return self.get(obj, 'position')

    def get_radius(self, obj):
        return self.option.get_size(self.get_size(obj))

    def get_3d_coords(self, obj):
        return self.get(obj, '3d_coords')

    def get_material(self, obj):
        return self.get(obj, 'material')

    def get_color(self, obj):
        return self.get(obj, 'color')

    def is_visible(self, obj):
        position = self.get_position(obj)
        return self.option.is_visible(*position)
