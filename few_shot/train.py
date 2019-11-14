#!/usr/bin/env python3
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import os
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split

from few_shot.data.omniglot_data_loader import omni_data_generator
from few_shot.data.mnist_data_loader import mnist_data_generator
from few_shot.models.simple_cnn import get_simple_cnn
# from few_shot.models.pre_trained_cnn import get_pretrained_model
from few_shot.utils import recall, precision, f1

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
CWD = os.path.dirname(os.path.realpath(__file__))
CHECKPOINTS_DIR = os.path.join(CWD, 'model_checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def train(dataset, image_shape, batch_size):
    '''
    Trains the model
    '''
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
        train_path = os.path.join(CWD, 'data/images_background')
        validation_path = os.path.join(CWD, 'data/images_evaluation')
        training_generator = omni_data_generator(train_path, batch_size,
                                                 image_shape)
        validation_generator = omni_data_generator(validation_path, batch_size,
                                                   image_shape)
        num_training_samples = 450
        num_validation_samples = 300

    model_name = '{}_basic_cnn'.format(dataset)
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

    model = get_simple_cnn(image_shape)

    model.summary()

    # opt = Adam(lr=1.0e-5)
    opt = SGD(lr=0.005, clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc', f1, precision, recall])

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(num_training_samples // batch_size),
        epochs=350,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=int(num_validation_samples // batch_size),
        shuffle=True)


if __name__ == '__main__':
    # train('mnist', (32, 32, 3), 64)
    train('omniglot', (32, 32, 3), 48)
