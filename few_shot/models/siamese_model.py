#!/usr/bin/env python3
import keras.backend as keras_backend
from keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import (
    Dense,
    Lambda,
    Input,
    Flatten,
    Convolution2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
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


def get_dense_encoder(input_shape, weights):
    '''
    Return the Dense architecture to represent the siamese branch
    '''
    model = DenseNet121(weights=weights, input_shape=input_shape, include_top=False)
    for layer in model.layers:
        layer.trainable = True
    return model


def get_resnet_encoder(input_shape, weights):
    '''
    Return the Dense architecture to represent the siamese branch
    '''
    if not weights:
        print('NOT using weights')
    model = ResNet50(weights=weights, input_shape=input_shape, include_top=False)
    for i, layer in enumerate(model.layers):
        if i < 19:
            layer.trainable = False
    return model


def get_lenet_encoder(input_shape):
    '''
    Return the LeNet architecture to represent the siamese branch
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
    x = Convolution2D(128,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Convolution2D(128,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    return Model(input=input_layer, outputs=x)


def get_siamese_model(input_shape, encoder='dense', weights=None):
    '''
    Returns the CNN model
    '''
    input_a = Input(shape=input_shape)
    # x = Dropout(0.1)(input_a)
    input_b = Input(shape=input_shape)
    # y = Dropout(0.1)(input_b)

    # instatiating the model so that it becomes shared weights
    if encoder == 'lenet':
        shared_cnn_encoder = get_lenet_encoder(input_shape)
    elif encoder == 'dense':
        shared_cnn_encoder = get_dense_encoder(input_shape, weights=weights)
    elif encoder == 'resnet':
        shared_cnn_encoder = get_resnet_encoder(input_shape, weights=weights)
    else:
        raise Exception('Failed to specify a valid encoder with: {}'
                        .format(encoder))

    encoded_a = shared_cnn_encoder(input_a)
    encoded_b = shared_cnn_encoder(input_b)

    merge_layer = Lambda(euclidean_distance)([encoded_a, encoded_b])
    merge_layer = Flatten()(merge_layer)
    x = Dense(512, activation="relu")(merge_layer)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_a, input_b], outputs=x)

    return model
