import argparse
from pathlib import Path

import h5py
import jsonlines
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


FLUSH = 100


def get_group(h5, name):
    if name not in h5:
        h5.create_group(name)
    return h5[name]


def write_h5(h5, split, samples, image_dir, image_width, image_height):
    f_split = get_group(h5, split)
    f_image = get_group(f_split, 'image')
    f_data = get_group(f_split, 'data')

    images = []
    keys = []
    basic_keys = []
    for sample in tqdm(samples, ncols=80, desc=f'{split} data'):
        name = str(sample['idx'])
        keys.append(name)
        if len(sample['transformations']) == 1:
            basic_keys.append(name)
        f_data.create_dataset(name, data=json.dumps(sample))
        images.extend(
            list(sample['states'][0]['images'].values()) 
            + list(sample['states'][-1]['images'].values()))
    f_split.create_dataset('keys', data=str(keys))
    f_split.create_dataset('basic_keys', data=str(basic_keys))

    for image in tqdm(images, desc=f'{split} images'):
        image_path = image_dir / image
        assert image_path.is_file()
        image_array = read_image(image_path, image_width, image_height)
        f_image.create_dataset(image, data=image_array, dtype=np.uint8)


def read_image(path, resize_w, resize_h):
    img = Image.open(path)
    if img.size != (resize_w, resize_h):
        img = img.resize((resize_w, resize_h))
    img = np.array(img)
    return img


def preprocess(
        directory, force=False, split=[500000, 10000, 20000],
        split_name=['train', 'val', 'test'], width=160, height=120):
    assert len(split) == len(split_name)
    assert split.count(-1) <= 1

    root = Path(directory)
    data_path = root / 'data.jsonl'
    image_dir = root / 'image'
    h5_path = root / 'data.h5'

    if h5_path.exists() and not force:
        print(f'Found existing {h5_path}. Use --force to overwrite.')
        return

    with jsonlines.open(data_path, 'r') as f:
        samples = list(f)
        n_sample = len(samples)
    print(f'Total samples: {n_sample}.')

    if -1 in split:
        assert sum(split) + 1 <= n_sample
        split[split.index(-1)] = n_sample - (sum(split) + 1)
    else:
        n_sample = sum(split)
        assert n_sample <= len(samples)

    print('\nsplits:')
    for s, n in zip(split, split_name):
        print(f'- {s}: {n}')
    print('\n')

    with h5py.File(h5_path, 'w', libver='latest') as f:
        start_idx = 0
        for s, n in zip(split_name, split):
            split_samples = samples[start_idx : start_idx + n]
            start_idx += n
            write_h5(f, s, split_samples, image_dir, width, height)


def preprocess_parser(parser):
    parser.add_argument('dataroot')
    parser.add_argument('--width', default=160)
    parser.add_argument('--height', default=120)
    parser.add_argument(
        '--split-name', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--split', nargs='+', default=[500000, 10000, 20000])
    parser.add_argument('--force', action='store_true')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    preprocess_parser(parser)

    args = parser.parse_args()
    preprocess(
        args.dataroot, args.force, args.split, args.split_name, args.width,
        args.height)