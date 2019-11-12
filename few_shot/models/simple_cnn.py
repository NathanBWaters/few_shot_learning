import keras.backend as keras_backend
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Lambda,
    Input,
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


def get_siamese_branch(input_shape):
    '''
    Return the siamese branch
    '''
    branch = Sequential(input_shape=input_shape)
    branch.add(
        Convolution2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001)))
    branch.add(BatchNormalization())
    branch.add(
        Convolution2D(32,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001)))
    branch.add(BatchNormalization())
    branch.add(
        Convolution2D(64,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001)))
    branch.add(BatchNormalization())
    branch.add(
        Convolution2D(64,
                      kernel_size=(3, 3),
                      activation='relu',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(0.001)))
    branch.add(BatchNormalization())

    return branch


def get_model(input_shape):
    '''
    Returns the CNN model
    '''
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    encoded_a = get_siamese_branch(input_a)
    encoded_b = get_siamese_branch(input_b)

    merge_layer = Lambda(euclidean_distance)([encoded_a, encoded_b])
    output_layer = Dense(1, activation="sigmoid")(merge_layer)
    model = Model(inputs=[input_a, input_b], outputs=output_layer)

    return model
