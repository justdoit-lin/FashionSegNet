from keras import layers
from keras.layers import (Activation, BatchNormalization, Conv2D, Input,
                          MaxPooling2D, ZeroPadding2D)
from keras.initializers import random_normal


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=1):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_initializer = random_normal(stddev=0.02), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate = dilation_rate, kernel_initializer = random_normal(stddev=0.02), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer = random_normal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate=1):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer = random_normal(stddev=0.02), 
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate = dilation_rate, kernel_initializer = random_normal(stddev=0.02), 
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer = random_normal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer = random_normal(stddev=0.02), 
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
    
def get_resnet50_encoder(inputs_size):
    block4_dilation = 1
    block5_dilation = 2
    block4_stride = 2
    
    img_input = Input(shape=inputs_size)

    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer = random_normal(stddev=0.02), name='conv1', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1), name='conv2_pad')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer = random_normal(stddev=0.02), name='conv2', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv2')(x)
    x = Activation(activation='relu')(x)

    x = ZeroPadding2D(padding=(1, 1), name='conv3_pad')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_initializer = random_normal(stddev=0.02), name='conv3', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv3')(x)
    x = Activation(activation='relu')(x)
    f1 = x

    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = x

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(block4_stride,block4_stride))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation_rate=block4_dilation)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1,1), dilation_rate=block4_dilation)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=block5_dilation)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=block5_dilation)
    f4 = x 

    return img_input, f1, f2, f3, f4

