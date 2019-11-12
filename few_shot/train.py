#!/usr/bin/env python3
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import os
import wandb
from wandb.keras import WandbCallback

from few_shot.data_loader import data_generator
from few_shot.models.simple_cnn import get_simple_cnn


BATCH_SIZE = 64
IMAGE_SIZE = (28, 28, 1)
CWD = os.path.dirname(os.path.realpath(__file__))


def train():
    '''
    Trains the model
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    training_generator = data_generator(x_train, y_train, BATCH_SIZE)
    validation_generator = data_generator(x_test, y_test, BATCH_SIZE)
    # test_generator = data_generator(x_test, y_test)

    num_training_samples = len(y_train)
    num_validation_samples = len(x_test)

    wandb.init(name='mnist_simple_cnn', project='few_shot')
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            mode='auto',
            min_delta=1.0e-5,
            cooldown=0,
            min_lr=0.000001),
        WandbCallback(),
        ModelCheckpoint(os.path.join(CWD, 'models/basic_cnn_{epoch}.h5'),
                        period=1)
    ]

    model = get_simple_cnn(IMAGE_SIZE)

    model.summary()

    # opt = Adam(lr=1.0e-5)
    opt = SGD(lr=0.005, clipvalue=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(num_training_samples // BATCH_SIZE),
        epochs=200,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=int(num_validation_samples // BATCH_SIZE),
        workers=8,
        max_queue_size=250,
        use_multiprocessing=False,
        shuffle=True
    )


if __name__ == '__main__':
    train()
