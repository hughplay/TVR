# -*- coding: utf-8 -*-
import hashlib
import time
from PIL import Image


def count_color(image_path):
    img = Image.open(image_path)
    colors = img.getcolors()
    img.close()
    return colors


def time2str(time_used):
    gaps = [
        ('days', 86400000),
        ('h', 3600000),
        ('min', 60000),
        ('s', 1000),
        ('ms', 1)
    ]
    time_used *= 1000
    time_str = []
    for unit, gap in gaps:
        val = time_used // gap
        if val > 0:
            time_str.append('{}{}'.format(int(val), unit))
            time_used -= val * gap
    if len(time_str) == 0:
        time_str.append('0ms')
    return ' '.join(time_str)


def get_time(t=None):
    if t is None:
        t = time.time()
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def model_size(model):
    count = sum(p.numel() for p in model.parameters())
    if count < (2**20):
        return '%.1fK' % (count / (2**10))
    else:
        return '%.1fM' % (count / (2**20))


def hash_seed(*items, width=32):
    # width: range of seed: [0, 2**width)
    sha = hashlib.sha256()
    for item in items:
        sha.update(str(item).encode('utf-8'))
    return int(sha.hexdigest()[23:23+width//4], 16)