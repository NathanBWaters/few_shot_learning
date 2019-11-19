#!/usr/bin/env python3
import numpy as np
import random
import os
import cv2
import pickle
from sklearn.utils import shuffle
from PIL import Image
import imgaug.augmenters as iaa

OMNIGLOT_FILE = 'omniglot.p'


def resize_omni(image, image_shape):
    '''
    Resizes an mnist image so it will work with a pre-trained network
    '''
    try:
        return np.array(Image.fromarray(image).resize((image_shape[0], image_shape[1])))
    except ValueError as e:
        import pdb; pdb.set_trace()
        print('hmm')


def augment_pair(image_a, image_b):
    '''
    Augments a pair of input images in the same way
    '''
    if (random.uniform(0, 1) > 0.5):
        image_a = np.fliplr(image_a)
        image_b = np.fliplr(image_b)

    if (random.uniform(0, 1) > 0.5):
        image_a = np.flipud(image_a)
        image_b = np.flipud(image_b)

    # adding distorion
    # seq = iaa.Sequential([
    #     iaa.Crop(percent=(0, 0.1)),  # random crops
    #     # Small gaussian blur with random sigma between 0 and 0.5.
    #     # But we only blur about 50% of all images.
    #     iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    #     # Strengthen or weaken the contrast in each image.
    #     iaa.ContrastNormalization((0.75, 1.5)),
    #     # Add gaussian noise.
    #     # For 50% of all images, we sample the noise once per pixel.
    #     # For the other 50% of all images, we sample the noise per pixel AND
    #     # channel. This can change the color (not only brightness) of the
    #     # pixels.
    #     iaa.AdditiveGaussianNoise(loc=0,
    #                               scale=(0.0, 0.05*255),
    #                               per_channel=0.5),
    #     # Make some images brighter and some darker.
    #     # In 20% of all cases, we sample the multiplier once per channel,
    #     # which can end up changing the color of the images.
    #     iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #     # Apply affine transformations to each image.
    #     # Scale/zoom them, translate/move them, rotate them and shear them.
    #     iaa.Affine(
    #         scale={"x": (0.8, 1.05), "y": (0.8, 1.05)},
    #         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #         rotate=(-5, 5),
    #         shear=(-4, 4),
    #         mode='edge',
    #     )
    # ], random_order=True)  # apply augmenters in random order
    # image_a, image_b = seq(images=[image_a, image_b])

    return image_a, image_b


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


def make_pairs_batch(omni_dict, batch_size, image_shape, augment=False):
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

        if augment:
            original_feature, matching_feature = augment_pair(
                original_feature, matching_feature)

        original_feature = resize_omni(original_feature, image_shape)
        matching_feature = resize_omni(matching_feature, image_shape)
        pairs.append([original_feature, matching_feature])
        pair_labels.append(0)

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

        if augment:
            original_feature, non_matching_feature = augment_pair(
                original_feature, non_matching_feature)

        original_feature = resize_omni(original_feature, image_shape)
        non_matching_feature = resize_omni(non_matching_feature, image_shape)
        pairs.append([original_feature, non_matching_feature])
        pair_labels.append(1)

    pairs, pair_labels = shuffle(pairs, pair_labels)

    pairs = np.array(pairs)

    return [pairs[:, 0], pairs[:, 1]], np.array(pair_labels)


def omni_data_generator(dataset_path, batch_size, image_shape, augment=False):
    '''
    Generator that yields a batch
    '''
    omni_dict = get_omni_dataset(dataset_path)
    while True:
        features, labels = make_pairs_batch(omni_dict,
                                            batch_size,
                                            image_shape,
                                            augment=augment)

        yield features, labels
