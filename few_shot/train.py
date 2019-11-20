#!/usr/bin/env python3
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import os
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as keras_backend

from few_shot.data.car_data_loader import get_car_data, car_generator
from few_shot.data.omniglot_data_loader import omni_data_generator
from few_shot.data.mnist_data_loader import mnist_data_generator
from few_shot.models.siamese_model import get_siamese_model
# from few_shot.models.pre_trained_cnn import get_pretrained_model
from few_shot.utils import recall, precision, f1
from few_shot.constants import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    OMNIGLOT_SHAPE,
    CAR_SHAPE,
    CAR_BATCH_SIZE
)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = keras_backend.square(y_pred)
    margin_square = keras_backend.square(keras_backend.maximum(margin - y_pred, 0))
    return keras_backend.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def acc(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return keras_backend.mean(keras_backend.equal(
        y_true, keras_backend.cast(y_pred < 0.5, y_true.dtype)))


def train(dataset, image_shape, batch_size):
    '''
    Trains the model
    '''
    if dataset == 'cars':
        train, validation, _ = get_car_data()
        training_generator = car_generator(train, batch_size, image_shape)
        validation_generator = car_generator(
            validation, batch_size, image_shape)
        num_training_samples = len(train)
        num_validation_samples = len(validation)

    if dataset == 'mnist':
        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.mnist.load_data()
        x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                        y_test,
                                                        test_size=0.5)

        training_generator = mnist_data_generator(x_train, y_train, batch_size,
                                                  image_shape)
        validation_generator = mnist_data_generator(x_val, y_val, batch_size,
                                                    image_shape)

        num_training_samples = len(x_train)
        num_validation_samples = len(x_val)

    if dataset == 'omniglot':
        train_path = os.path.join(DATA_DIR, 'images_background')
        validation_path = os.path.join(DATA_DIR, 'images_evaluation')
        training_generator = omni_data_generator(
            train_path,
            batch_size,
            image_shape,
            augment=True)
        validation_generator = omni_data_generator(validation_path, batch_size,
                                                   image_shape)
        num_training_samples = 450
        num_validation_samples = 300

    encoder = 'resnet'
    model_name = '{}_{}_pretrained_augmentation'.format(dataset, encoder)
    model = get_siamese_model(image_shape, encoder=encoder, weights='imagenet')

    wandb.init(name=model_name, project='few_shot')
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=10,
                          verbose=1,
                          mode='auto',
                          min_delta=1.0e-5,
                          cooldown=0,
                          min_lr=0.00001),
        WandbCallback(),
        ModelCheckpoint(os.path.join(CHECKPOINTS_DIR,
                                     model_name + '_{epoch}.h5'),
                        period=5)
    ]

    model.summary()

    opt = Adam(lr=3.0e-4)
    # opt = SGD(lr=0.005, clipvalue=0.5)
    model.compile(loss=contrastive_loss,
                  optimizer=opt,
                  metrics=[acc, f1, precision, recall])

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(num_training_samples // batch_size),
        epochs=350,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=int(num_validation_samples // batch_size),
        shuffle=True)

    model.save(os.path.join('model_checkpoints', model_name))


if __name__ == '__main__':
    # train('mnist', (64, 64, 3), 64)
    # train('omniglot', OMNIGLOT_SHAPE, 48)
    train('cars', CAR_SHAPE, CAR_BATCH_SIZE)
