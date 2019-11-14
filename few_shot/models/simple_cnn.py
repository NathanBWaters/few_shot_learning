#!/usr/bin/env python3
import keras.backend as keras_backend
from keras.models import Model
from keras.layers import (
    Dense,
    Lambda,
    Input,
    Flatten,
    Convolution2D,
    BatchNormalization,
)
from keras.regularizers import l2


def euclidean_distance(vects):
    '''
    Given two vectors, return the euclidean distance
    '''
    x, y = vects
    sum_square = keras_backend.sum(keras_backend.square(x - y),
                                   axis=1,
                                   keepdims=True)
    return keras_backend.sqrt(
        keras_backend.maximum(sum_square, keras_backend.epsilon()))


def get_shared_encoder(input_shape):
    '''
    Return the siamese branch
    '''
    input_layer = Input(shape=input_shape)
    x = Convolution2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(input_layer)
    x = BatchNormalization()(x)
    x = Convolution2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    return Model(input=input_layer, outputs=x)


def get_simple_cnn(input_shape):
    '''
    Returns the CNN model
    '''
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # instatiating the model so that it becomes shared weights
    shared_cnn_encoder = get_shared_encoder(input_shape)

    encoded_a = shared_cnn_encoder(input_a)
    encoded_b = shared_cnn_encoder(input_b)

    merge_layer = Lambda(euclidean_distance)([encoded_a, encoded_b])
    merge_layer = Flatten()(merge_layer)
    x = Dense(100, activation="relu")(merge_layer)
    x = Dense(25, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_a, input_b], outputs=x)

    return model
