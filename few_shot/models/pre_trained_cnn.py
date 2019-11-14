#!/usr/bin/env python3
import keras.backend as keras_backend
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
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
    model = MobileNetV2(input_shape=input_shape,
                        alpha=1.0,
                        include_top=False,
                        weights='imagenet',
                        pooling=None)

    for layer in model.layers:
        layer.trainable = False

    return model


def get_pretrained_model(input_shape):
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
    output_layer = Dense(1, activation="sigmoid")(merge_layer)
    model = Model(inputs=[input_a, input_b], outputs=output_layer)

    return model
