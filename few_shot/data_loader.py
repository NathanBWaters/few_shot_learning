import tensorflow as tf
import numpy as np
import random

BATCH_SIZE = 64


def make_pairs_batch(features, labels, batch_size):
    '''
    Given the features and labels of a dataset, return a batch of matching and
    not matching pairs
    '''
    num_classes = max(labels) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    pair_labels = []

    for i in range(batch_size):
        # select a random feature
        index = random.randint(0, len(features) - 1)

        # adding a matching example
        original_feature = features[index]
        feature_label = y[index]
        matching_index = random.choice(digit_indices[feature_label])
        matching_feature = features[matching_index]

        pairs.append([original_feature, matching_feature])
        pair_labels.append(1)

        # adding a non-matching example
        non_matching_label = random.randint(0, num_classes - 1)
        while feature_label == non_matching_label:
            non_matching_label = random.randint(0, num_classes - 1)

        non_matching_feature = features[random.choice(
            digit_indices[non_matching_label])]

        pairs.append([original_feature, non_matching_feature])
        pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)


def data_generator(data_features, data_labels, batch_size=BATCH_SIZE):
    while True:
        features, labels = make_pairs_batch(data_features,
                                            data_labels,
                                            batch_size)

        return features, labels
