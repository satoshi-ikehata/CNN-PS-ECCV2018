from keras import backend as K
from keras.models import Input, Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda
from keras.layers import Merge, merge, Concatenate, concatenate, MaxPooling1D, multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv1D, Conv2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam, Adadelta

def get_densenet_2d_channel_first_2dense(rows, cols):

    inputs1 = Input((1, rows, cols))

    x0 = inputs1
    x1 = Conv2D(16, (3, 3), padding='same', name='conv1')(x0)

    # 1st Denseblock
    x1a = Activation('relu')(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv2')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = concatenate([x2, x1], axis=3)

    xc1a = Activation('relu')(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv3')(xc1a)
    x3 = Dropout(0.2)(x3)

    xc2 = concatenate([x3,x2,x1], axis=3)

    # Transition
    xc2a = Activation('relu')(xc2)
    x4 = Conv2D(48, (1, 1), padding='same', name='conv4')(xc2a)
    x4 = Dropout(0.2)(x4)
    x1 = AveragePooling2D((2, 2), strides=(2, 2))(x4)

    # 2nd Dense block
    x1a = Activation('relu')(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv5')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = concatenate([x2, x1], axis=3)

    xc1a = Activation('relu')(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv6')(xc1a)
    x3 = Dropout(0.2)(x3)
    xc2 = concatenate([x3,x2,x1], axis=3)
    # xc2 = x3

    xc2a = Activation('relu')(xc2)
    x4 = Conv2D(80, (1, 1), padding='same', name='conv7')(xc2a)

    x = Flatten()(x4)
    x = Dense(128,activation='relu',name='dense1b')(x)
    x = Dense(3, name='dense2')(x)

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1))
    x = normalize(x)

    outputs = x

    model = Model(inputs = inputs1, outputs = outputs)
    return model

def get_densenet_2d_channel_last_2dense(rows, cols):

    inputs1 = Input((rows, cols, 1))

    x0 = inputs1
    x1 = Conv2D(16, (3, 3), padding='same', name='conv1')(x0)

    # 1st Denseblock
    x1a = Activation('relu')(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv2')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = concatenate([x2, x1], axis=3)

    xc1a = Activation('relu')(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv3')(xc1a)
    x3 = Dropout(0.2)(x3)

    xc2 = concatenate([x3,x2,x1], axis=3)

    # Transition
    xc2a = Activation('relu')(xc2)
    x4 = Conv2D(48, (1, 1), padding='same', name='conv4')(xc2a)
    x4 = Dropout(0.2)(x4)
    x1 = AveragePooling2D((2, 2), strides=(2, 2))(x4)

    # 2nd Dense block
    x1a = Activation('relu')(x1)
    x2 = Conv2D(16, (3, 3), padding='same', name='conv5')(x1a)
    x2 = Dropout(0.2)(x2)

    xc1 = concatenate([x2, x1], axis=3)

    xc1a = Activation('relu')(xc1)
    x3 = Conv2D(16, (3, 3), padding='same', name='conv6')(xc1a)
    x3 = Dropout(0.2)(x3)
    xc2 = concatenate([x3,x2,x1], axis=3)

    xc2a = Activation('relu')(xc2)
    x4 = Conv2D(80, (1, 1), padding='same', name='conv7')(xc2a)

    x = Flatten()(x4)
    x = Dense(128,activation='relu',name='dense1b')(x)
    x = Dense(3, name='dense2')(x)

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1))
    x = normalize(x)

    outputs = x

    model = Model(inputs = inputs1, outputs = outputs)
    return model
