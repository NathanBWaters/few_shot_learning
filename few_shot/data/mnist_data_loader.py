#!/usr/bin/env python3
import numpy as np
import random
from PIL import Image


def resize_mnist(image, image_shape):
    '''
    Resizes an mnist image so it will work with a pre-trained network
    '''
    image = np.stack((image, ) * 3, axis=-1)
    return np.array(Image.fromarray(image).resize((image_shape[0], image_shape[1])))


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
