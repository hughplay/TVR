# -*- coding: utf-8 -*-
import argparse
from collections.abc import Sequence
from pathlib import Path

from munch import Munch, munchify
from ruamel.yaml import YAML


class Config(Munch):

    def __init__(self, file):
        self._yaml_manager = YAML()
        self._yaml_manager.preserve_quotes = True

        self._file = file
        self._save_target = file
        self._yaml = self.file2yaml(self._file)

        self._update(self._yaml, True)

    def _update(self, _yaml, _all=False):
        _munch = munchify(_yaml)
        if _all:
            self.update(_munch)
        else:
            for k in self:
                if k in _munch:
                    self[k] = _munch[k]

    def file2yaml(self, file):
        file = Path(file)
        _yaml = self._yaml_manager.load(file)
        return _yaml

    def _update_yaml_node(self, munch_obj, yaml_obj):
        for key in munch_obj.keys():
            m_val = munch_obj[key]

            if key not in yaml_obj:
                if type(m_val) is Munch:
                    yaml_obj[key] = dict(m_val)
                else:
                    yaml_obj[key] = m_val
                return
            else:
                y_val = yaml_obj[key]

            if type(m_val) is Munch:
                self._update_yaml_node(m_val, y_val)
            elif isinstance(m_val, Sequence) and len(
                    m_val) > 0 and type(m_val[0]) is Munch:
                for i, item in enumerate(m_val):
                    if i < len(y_val):
                        self._update_yaml_node(m_val[i], y_val[i])
                    else:
                        y_val.append(m_val[i])
                if len(m_val) < len(y_val):
                    for i in range(len(m_val), len(y_val)):
                        y_val.pop(i)
            else:
                if m_val != y_val:
                    yaml_obj[key] = m_val

    def update_yaml(self):
        self._update_yaml_node(self, self._yaml)

    def save(self, file=None):
        self.update_yaml()
        file = self._save_target if file is None else file
        self._yaml_manager.dump(self._yaml, Path(file))

    def __setattr__(self, k, v):
        # Can only change the outermost attribute.
        try:
            if not k.startswith('_'):
                object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def extend_parser(self, parser):

        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError(
                    'Boolean value expected.')

        def parseNone(v):
            if v.lower() =='none' or v.lower() == 'null':
                return None
            try:
                return int(v)
            except:
                pass
            try:
                return float(v)
            except:
                pass
            try:
                return bool(v)
            except:
                pass
            return v

        assert type(parser) is argparse.ArgumentParser
        for k, v in self.items():
            if type(self[k]) is Munch:
                continue

            values = self._yaml.ca.items
            comment = values[k][2] if k in values else None
            if comment is not None:
                comment = comment.value.replace('#', '').strip()
            v_type = type(v)
            v_type = float if 'float' in str(v_type) else v_type
            v_type = str if 'str' in str(v_type) else v_type
            v_type = parseNone if 'None' in str(v_type) else v_type

            try:
                if v_type is bool:
                    parser.add_argument(
                        '--' + k, default=argparse.SUPPRESS, nargs='?',
                        type=str2bool, const=True, help=comment)
                else:
                    parser.add_argument(
                        '--' + k, default=argparse.SUPPRESS, type=v_type,
                        help=comment)
            except Exception:
                pass

    def merge_args(self, args):
        assert type(args) is argparse.Namespace
        self._update(args.__dict__)

    def merge_file(self, file):
        self._save_target = file
        _yaml = self.file2yaml(file)
        self._update(_yaml)