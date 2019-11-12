import tensorflow as tf
from keras.callbacks imoprt ModelCheckpoint, ReduceLROnPlateau
from few_shot.data_loader import data_generator

import wandb
from wandb.keras import WandbCallback

def train():
    '''
    Trains the model
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    training_generator = data_generator(x_train, y_train)
    validation_generator = data_generator(x_test, y_test)
    test_generator = data_generator(x_test, y_test)

    wandb.init()
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
    ]

if __name__ == '__main__':
    train()
