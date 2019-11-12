#!/usr/bin/env python3
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import os
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split

from few_shot.data_loader import data_generator
from few_shot.models.simple_cnn import get_simple_cnn
from few_shot.utils import recall, precision, f1

BATCH_SIZE = 64
IMAGE_SIZE = (28, 28, 1)
CWD = os.path.dirname(os.path.realpath(__file__))
CHECKPOINTS_DIR = os.path.join(CWD, 'model_checkpoints')
os.mkdirs(CHECKPOINTS_DIR, exist_ok=True)


def train():
    '''
    Trains the model
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    training_generator = data_generator(x_train, y_train, BATCH_SIZE, IMAGE_SIZE)
    validation_generator = data_generator(x_val, y_val, BATCH_SIZE, IMAGE_SIZE)
    # test_generator = data_generator(x_test, y_test, BATCH_SIZE, IMAGE_SIZE)

    num_training_samples = len(x_train)
    num_validation_samples = len(x_val)

    model_name = 'mnist_simple_cnn'
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
        ModelCheckpoint(
            os.path.join(CHECKPOINTS_DIR, model_name + '_{epoch}.h5'), period=1)
    ]

    model = get_simple_cnn(IMAGE_SIZE)

    model.summary()

    opt = Adam(lr=1.0e-5)
    opt = SGD(lr=0.005, clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc', f1, precision, recall])

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(num_training_samples // BATCH_SIZE),
        epochs=200,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=int(num_validation_samples // BATCH_SIZE),
        shuffle=True
    )


if __name__ == '__main__':
    train()
