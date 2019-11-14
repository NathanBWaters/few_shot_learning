#!/usr/bin/env python3
import numpy as np
import random
import os
import cv2
import pickle
from sklearn.utils import shuffle
from PIL import Image

OMNIGLOT_FILE = 'omniglot.p'


def resize_omni(image, image_shape):
    '''
    Resizes an mnist image so it will work with a pre-trained network
    '''
    return np.array(Image.fromarray(image).resize((image_shape[0], image_shape[1])))



def get_omni_dataset(dataset_path):
    '''
    path => Path of train directory or test directory
    '''
    cache_file = os.path.join(dataset_path, OMNIGLOT_FILE)
    if os.path.exists(cache_file):
        print('Returning cached omniglot data')
        omniglot_dict = pickle.load(open(cache_file, 'rb'))
        return omniglot_dict

    omniglot_dict = {}

    print('Caching omniglot data')
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(dataset_path):
        print("loading alphabet: " + alphabet)
        omniglot_dict[alphabet] = {}
        alphabet_path = os.path.join(dataset_path, alphabet)

        for letter in os.listdir(alphabet_path):
            omniglot_dict[alphabet][letter] = []
            letter_path = os.path.join(alphabet_path, letter)

            # read all the images in the current category
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = cv2.imread(image_path)
                omniglot_dict[alphabet][letter].append(image)

    pickle.dump(omniglot_dict, open(cache_file, 'wb'))
    return omniglot_dict


def make_pairs_batch(omni_dict, batch_size, image_shape):
    '''
    Given the features and labels of a dataset, return a batch of matching and
    not matching pairs
    '''
    pairs = []
    pair_labels = []

    class_imbalance_noise = random.randint(-7, 7)
    num_positive_pairs = batch_size // 2 + class_imbalance_noise
    num_negative_pairs = batch_size // 2 - class_imbalance_noise

    for i in range(num_positive_pairs):
        # select a random feature
        alphabet = random.choice(list(omni_dict.keys()))
        character = random.choice(list(omni_dict[alphabet].keys()))

        # adding a matching example
        original_feature = random.choice(omni_dict[alphabet][character])
        matching_feature = random.choice(omni_dict[alphabet][character])

        pairs.append([resize_omni(original_feature, image_shape),
                      resize_omni(matching_feature, image_shape)])
        pair_labels.append(1)

    for i in range(num_negative_pairs):
        alphabet = random.choice(list(omni_dict.keys()))
        character = random.choice(list(omni_dict[alphabet].keys()))

        # adding a matching example
        original_feature = random.choice(omni_dict[alphabet][character])

        # adding a non-matching example
        different_alphabet = random.choice(list(omni_dict.keys()))
        while alphabet == different_alphabet:
            different_alphabet = random.choice(list(omni_dict.keys()))

        # get a sample from a different class
        non_matching_character = random.choice(list(omni_dict[different_alphabet].keys()))
        non_matching_feature = random.choice(
            omni_dict[different_alphabet][non_matching_character])

        pairs.append([resize_omni(original_feature, image_shape),
                      resize_omni(non_matching_feature, image_shape)])
        pair_labels.append(0)

    pairs, pair_labels = shuffle(pairs, pair_labels)

    pairs = np.array(pairs)

    return [pairs[:, 0], pairs[:, 1]], np.array(pair_labels)


def omni_data_generator(dataset_path, batch_size, image_shape):
    '''
    Generator that yields a batch
    '''
    omni_dict = get_omni_dataset(dataset_path)
    while True:
        features, labels = make_pairs_batch(omni_dict,
                                            batch_size,
                                            image_shape)

        yield features, labels
