#!/usr/bin/env python3
import numpy as np
import random
import os
from scipy.io import loadmat
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import imgaug.augmenters as iaa

from few_shot.constants import DATA_DIR


def path_to_image(path, image_shape):
    '''
    Resizes an mnist image so it will work with a pre-trained network
    '''
    car_image = cv2.imread(path)
    car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
    return np.array(Image.fromarray(car_image).resize(
        (image_shape[0], image_shape[1])))


def augment_pair(image_a, image_b):
    '''
    Augments a pair of input images in the same way
    '''
    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # flip the image left to right half the time
        iaa.Fliplr(0.5),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0,
                                  scale=(0.0, 0.05*255),
                                  per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-5, 5),
            shear=(-4, 4),
            # mode='edge',
        ),
        iaa.PerspectiveTransform(scale=(0.01, 0.05)),
    ], random_order=True)  # apply augmenters in random order
    image_a, image_b = seq(images=[image_a, image_b])

    return image_a, image_b


def make_pairs_batch(dataset, batch_size, image_shape):
    '''
    Given the features and labels of a dataset, return a batch of matching and
    not matching pairs
    '''
    cars_dict = defaultdict(list)
    for car in dataset:
        cars_dict[car.class_id].append(car.filename)

    pairs = []
    pair_labels = []

    for i in range(batch_size // 2):
        # select a random feature
        class_id = random.choice(list(cars_dict.keys()))
        anchor_car_path = random.choice(list(cars_dict[class_id]))
        matching_car_path = random.choice(list(cars_dict[class_id]))

        anchor_car = path_to_image(anchor_car_path, image_shape)
        matching_car = path_to_image(matching_car_path, image_shape)

        augmented_anchor, augmented_match = augment_pair(
            anchor_car, matching_car)
        pairs.append([augmented_anchor, augmented_match])
        pair_labels.append(0)

        # adding a non-matching example
        non_matching_class_id = random.choice(list(cars_dict.keys()))
        while non_matching_class_id == class_id:
            non_matching_class_id = random.choice(list(cars_dict.keys()))

        # get a sample from a different class
        non_matching_car_path = random.choice(list(
            cars_dict[non_matching_class_id]))
        non_matching_car = path_to_image(non_matching_car_path, image_shape)

        augmented_anchor, augmented_non_match = augment_pair(
            anchor_car, non_matching_car)
        pairs.append([augmented_anchor, augmented_non_match])
        pair_labels.append(1)

    pairs, pair_labels = shuffle(pairs, pair_labels)
    pairs = np.array(pairs)

    return [pairs[:, 0], pairs[:, 1]], np.array(pair_labels)


class Car():
    '''
    Creates car instance
    '''
    def __init__(self, np_data, classnames):
        self._filename = np_data[0][0]

        self.lr_x = np_data[1][0][0]
        self.lr_y = np_data[2][0][0]
        self.ul_x = np_data[3][0][0]
        self.ul_x = np_data[4][0][0]
        self.class_id = np_data[5][0][0]
        self.class_name = classnames[self.class_id]

    @property
    def filename(self):
        return os.path.join(DATA_DIR, self._filename)

    def __repr__(self):
        '''
        String representation of the Car
        '''
        return '<Car {} | {} />'.format(self.class_id, self.filename)


def np_to_cars(numpy_dataset, classnames):
    '''
    Converts the raw numpy array to Car instances
    '''
    cars = []
    for sample in numpy_dataset['annotations'][0][:]:
        car = Car(sample, classnames)
        cars.append(car)

    return cars


def car_generator(dataset, batch_size, image_shape):
    '''
    Given a dataset of Car instances, yield matching and non-matching pairs
    of car instances
    '''
    while True:
        features, labels = make_pairs_batch(dataset, batch_size, image_shape)

        yield features, labels


def get_car_generators(batch_size, image_shape):
    '''
    Return the training and validation car generators
    '''
    train, validation, test = get_car_data()

    return (
        car_generator(train, batch_size, image_shape),
        car_generator(validation, batch_size, image_shape),
        car_generator(test, batch_size, image_shape))


def get_car_data():
    '''
    Extracts the car data
    '''
    metadata = loadmat(os.path.join(DATA_DIR, 'car_data', 'cars_meta.mat'))
    classnames = {i + 1: a[0] for i, a in enumerate(metadata['class_names'][0])}

    car_data_raw = loadmat(os.path.join(DATA_DIR, 'car_data', 'cars_annos.mat'))

    random.seed(9001)
    car_data = np_to_cars(car_data_raw, classnames)
    random.shuffle(car_data)

    train, validation = train_test_split(car_data, test_size=0.3)
    validation, test = train_test_split(validation, test_size=0.5)
    return train, validation, test
