# ResNet50
# First thing first, import all the neccesary modules

import keras.backend as K
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

from convolutionalBlock import conv_block
from identityBlock import identity

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def ResNet50(input_shape=(64, 64, 2), classes=2):
    '''
        if the input shape is different, just provide the shape during function
        call. the number of class may vary depending on the problem set
    '''
    # the input to the keras model
    X_input = Input(shape=input_shape)
    # the output to the keras model
    X = ZeroPadding2D(padding=(3, 3))(X_input)
    # now the first convolution using 7x7 filter
    X = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), name='conv_1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # The first block one conv_block two identity blocks
    X = conv_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)(X)
    X = identity(X, f=3, filters=[64, 64, 256], stage=2, block='b')(X)
    X = identity(X, f=3, filters=[64, 64, 256], stage=2, block='c')(X)

    # second block
    X = conv_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)(X)
    X = identity(X, f=3, filters=[128, 128, 512], stage=3, block='b', s=2)(X)
    X = identity(X, f=3, filters=[128, 128, 512], stage=3, block='b', s=2)(X)
    X = identity(X, f=3, filters=[128, 128, 512], stage=3, block='b', s=2)(X)

    # third block
    X = conv_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)(X)
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='b', s=2)(X)
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='c', s=2)(X)
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='d', s=2)(X)
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='e', s=2)(X)
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='f', s=2)(X)

    # fourth block
    X = conv_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)(X)
    X = identity(X, f=3, filters=[512, 512, 2048], stage=5, block='b', s=2)(X)
    X = identity(X, f=3, filters=[512, 512, 2048], stage=5, block='c', s=2)(X)

    X = AveragePooling2D(pool_size=(2, 2), name='average_pool')(X)

    # flatten X for the fully connected layer
    X = Flatten()(X)
    X = Dense(units=classes, activation='softmax', name='fc_{}'.format(classes))

    # finally cleate a model using X_input and X
    model = Model(input=X_input, outputs=X, name='ResNet50')
    return model
