# -*- coding: utf-8 -*-
import random

import numpy as np


def balanced_sample(freq, exclude=[], tolerance=0.1, top=0, verbose=False):
    if top > 0:
        freq = {k: v for k, v in sorted(
                freq.items(), key=lambda item: (item[1], item[0]))[:top]}
    # exclude keys
    valid_keys = {
        k: v for k, v in freq.items() if k not in exclude}
    if len(valid_keys) == 0:
        return None
    # find the maximum count
    n_max = max(valid_keys.values())
    # minus the maximum count to normalize counts
    normalized_counts = [
        (k, n_max - v + tolerance) for k, v in sorted(valid_keys.items())]
    # compute the probability
    keys, counts = zip(*normalized_counts)
    prob = np.array(counts) / sum(counts)
    key_selected = np.random.choice(keys, p=prob)

    if verbose:
        print('---')
        for key, p in zip(keys, prob):
            print('- {}: {}'.format(key, p))
        print('selected: {}'.format(key_selected))

    return key_selected