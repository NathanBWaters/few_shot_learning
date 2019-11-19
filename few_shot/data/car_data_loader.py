#!/usr/bin/env python3
import numpy as np
import random
import os
from PIL import Image
from functools import partial
from pathlib import Path
from scipy.io import loadmat
# import tables

from few_shot.constants import CWD


def make_pairs_batch(features, labels, batch_size, image_shape):
    '''
    Given the features and labels of a dataset, return a batch of matching and
    not matching pairs
    '''
    num_classes = max(labels) + 1
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    pairs = []
    pair_labels = []

    for i in range(batch_size // 2):
        # select a random feature
        index = random.randint(0, len(features) - 1)

        # adding a matching example
        original_feature = features[index]
        feature_label = labels[index]

        # get a sample from the same class
        same_class_index = random.choice(digit_indices[feature_label])
        same_class_feature = features[same_class_index]

        pairs.append([resize_mnist(original_feature, image_shape),
                      resize_mnist(same_class_feature, image_shape)])
        pair_labels.append(1)

        # adding a non-matching example
        non_matching_label = random.randint(0, num_classes - 1)
        while feature_label == non_matching_label:
            non_matching_label = random.randint(0, num_classes - 1)

        # get a sample from a different class
        non_matching_feature = features[random.choice(
            digit_indices[non_matching_label])]

        pairs.append([resize_mnist(original_feature, image_shape),
                      resize_mnist(non_matching_feature, image_shape)])
        pair_labels.append(0)

    pairs = np.array(pairs)

    return [pairs[:, 0], pairs[:, 1]], np.array(pair_labels)


def mnist_data_generator(data_features, data_labels, batch_size, image_shape):
    '''
    Generator that yields a batch
    '''
    while True:
        features, labels = make_pairs_batch(data_features,
                                            data_labels,
                                            batch_size,
                                            image_shape)

        yield features, labels


def get_car_generators():
    '''
    Return the training and validation car generators
    '''
    meta = loadmat(os.path.join(CWD, 'data\\car_devkit\\cars_annos.mat'))
    import pdb; pdb.set_trace()
    # train_annos = loadmat(devkit_path/'cars_train_annos.mat')
    # test_annos = loadmat(devkit_path/'cars_test_annos.mat')

    labels = [c for c in meta['class_names'][0]]
    labels = pd.DataFrame(labels, columns=['labels'])
