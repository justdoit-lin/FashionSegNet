import tensorflow as tf
from keras.initializers import random_normal
from keras.layers import *
from keras.models import *
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, DepthwiseConv2D, Dropout, SeparableConv2D,
                          GlobalAveragePooling2D, Input, Lambda, Reshape,
                          Softmax, ZeroPadding2D)
from keras import backend as K 


from resnet50 import get_resnet50_encoder


def Swish(args):
    return (K.sigmoid(args) * args)

def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def resize_images(args):
    x = args[0]
    y = args[1]
    return tf.image.resize_images(x, (K.int_shape(y)[1], K.int_shape(y)[2]), align_corners=True)

def Sigmoid(args):
    return tf.keras.activations.sigmoid(args)

    #--------------------------------------------------------------#
    #	MSPP
    #--------------------------------------------------------------#
    
def pooling_branch(feats, pool_factor, out_channel, prefix):
    pool_size = strides = [pool_factor,pool_factor]
    x = AveragePooling2D(pool_size, strides=strides, padding='same', name = prefix+'_avgpool')(feats)
    x = Conv2D(out_channel//4, (1 ,1), kernel_initializer = random_normal(stddev=0.02), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(resize_images)([x, feats])
    return x

def extraction_branch(x, filters, prefix, rate, stride = 2, kernel_size = 3, epsilon = 1e-3):
    if rate == 2:
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv1')(x)
        x_out = BatchNormalization(name=prefix + '_res_BN1', epsilon=epsilon)(x_out)
        x_out = Lambda(Sigmoid)(x_out)
        x_out = Lambda(resize_images)([x_out, x])
        x_out = Multiply()([x_out, x])
        x_out = Conv2D(filters, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False, name=prefix + '_res_conv')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN3', epsilon=epsilon)(x_out)
        x = Lambda(Swish)(x_out)
        
    if rate == 4:
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv1')(x)
        x_out = BatchNormalization(name=prefix + '_res_BN1', epsilon=epsilon)(x_out)
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv2')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN2', epsilon=epsilon)(x_out)
        x_out = Lambda(Sigmoid)(x_out)
        x_out = Lambda(resize_images)([x_out, x])
        x_out = Multiply()([x_out, x])
        x_out = Conv2D(filters, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False, name=prefix + '_res_conv')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN3', epsilon=epsilon)(x_out)
        x = Lambda(Swish)(x_out)
        
    if rate == 8:
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv1')(x)
        x_out = BatchNormalization(name=prefix + '_res_BN1', epsilon=epsilon)(x_out)
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv2')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN2', epsilon=epsilon)(x_out)
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv3')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN3', epsilon=epsilon)(x_out)
        x_out = Lambda(Sigmoid)(x_out)
        x_out = Lambda(resize_images)([x_out, x])
        x_out = Multiply()([x_out, x])
        x_out = Conv2D(filters, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False, name=prefix + '_res_conv')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN4', epsilon=epsilon)(x_out)
        x = Lambda(Swish)(x_out)
        
    if rate == 16:
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv1')(x)
        x_out = BatchNormalization(name=prefix + '_res_BN1', epsilon=epsilon)(x_out)
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv2')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN2', epsilon=epsilon)(x_out)
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv3')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN3', epsilon=epsilon)(x_out)
        x_out = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), activation='relu', padding='same', use_bias=False, name=prefix + '_res_depconv4')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN4', epsilon=epsilon)(x_out)
        x_out = Lambda(Sigmoid)(x_out)
        x_out = Lambda(resize_images)([x_out, x])
        x_out = Multiply()([x_out, x])
        x_out = Conv2D(filters, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False, name=prefix + '_res_conv')(x_out)
        x_out = BatchNormalization(name=prefix + '_res_BN5', epsilon=epsilon)(x_out)
        x = Lambda(Swish)(x_out)
        
    return x

def dilation_branch(x, filters, prefix, kernel_size=3, rate=1, epsilon=1e-3):
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), dilation_rate=(rate, rate),
                        padding='same', use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    x = Activation('relu')(x)

    return x

def MixdSPP(x, out_channel, rate, atrous_rate, pool_rate, prefix):
    x_out = Conv2D(out_channel, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False)(x)
    x_out = BatchNormalization()(x_out)
    x_pool = pooling_branch(x_out, pool_rate, out_channel, prefix)
    x_atrous = dilation_branch(x_out, out_channel, prefix, atrous_rate)
    x_resize = extraction_branch(x_out, out_channel, prefix, rate)
    x_out = Concatenate(axis=-1)([x_pool, x_atrous])
    x_out = Conv2D(out_channel, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False)(x_out)
    x_out = BatchNormalization()(x_out)
    x_out = Multiply()([x_out, x_resize])
    x_out = Lambda(Swish)(x_out)
    return x_out

    #--------------------------------------------------------------#
    #	AGE
    #--------------------------------------------------------------#
    
def AGE(age_in, channel):
    age = Conv2D(channel, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False)(age_in)
    age = BatchNormalization()(age)
    x_shortout = age
    age = Conv2D(channel, kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(age)
    age = BatchNormalization()(age)
    age = DepthwiseConv2D(kernel_size=(30,30), strides = 1, activation='relu', padding='same', use_bias=False)(age)
    age = BatchNormalization()(age)
    age = Add()([age, x_shortout])
    age_out = Activation('relu')(age)
    return age_out
    
    #--------------------------------------------------------------#
    #	SIE
    #--------------------------------------------------------------#
    
def SIE(sie_in, channel):
    sie_conv = Conv2D(channel, kernel_size=(3,3), strides=(2,2), padding='same')(sie_in)
    sie_conv = BatchNormalization()(sie_conv)
    sie_conv = Activation('relu')(sie_conv)
    sie_ap = AveragePooling2D([2,2], strides=[2,2], padding='same')(sie_in)
    sie_out = Concatenate(axis=-1)([sie_conv, sie_ap])
    return sie_out

    #--------------------------------------------------------------#
    #	CIE
    #--------------------------------------------------------------#
    
def CIE(cie_in, stage, sie_out, channel):
    cie = Conv2D(channel, kernel_size=(1,1), strides = 1, padding='same', use_bias=False)(cie_in)
    cie = BatchNormalization()(cie)
    cie = GlobalAveragePooling2D()(cie)
    cie = Lambda(Sigmoid)(cie)
    cie = Reshape((1, 1, channel))(cie)
    cie = Multiply()([cie, stage])
    cie_out = Add()([cie, sie_out])
    return cie_out
  
    #--------------------------------------------------------------#
    #	Model
    #--------------------------------------------------------------#
    
def fashionsegnet(input_shape, num_classes):
    img_input, f1, f2, f3, f4 = get_resnet50_encoder(input_shape)
    out_channel = 2048
    
    #--------------------------------------------------------------#
    #	SIE
    #--------------------------------------------------------------#
    
    f1_SIE = SIE(f1, out_channel//16)
    f2_CAT = Concatenate(axis=-1)([f2, f1_SIE])
    f2_SIE = SIE(f2, out_channel//8)
    f2_CAT_SIE = SIE(f2_CAT, out_channel//4)
    f3_CAT = Concatenate(axis=-1)([f3, f2_SIE, f2_CAT_SIE])
    
    rates = [2, 4, 8, 16]
    pool_rates = [5, 10, 15, 30]
    atrous_rates = [3, 6, 12, 18]
    f4_Mix_names = ['f4_mspp1', 'f4_mspp2', 'f4_mspp3', 'f4_mspp4']
    f4_mspp_outs = [f4]
    f4_conv = Conv2D(out_channel//4, kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(f4)
    f4_conv = BatchNormalization()(f4_conv)
    f4_mspp_outs.append(f4_conv)
    for i in range(0, 4):
        f4_mspped = MixdSPP(f4, out_channel//4, prefix = f4_Mix_names[i], rate = rates[i], atrous_rate = atrous_rates[i], pool_rate = pool_rates[i])
        f4_mspp_outs.append(f4_mspped)
    f4_out = Concatenate(axis=-1)(f4_mspp_outs)
    f4_out = Conv2D(out_channel//4, kernel_size=(1,1), strides = 1, activation='relu', padding='same', use_bias=False)(f4_out)
    MSPP_out = BatchNormalization()(f4_out)
    
    #--------------------------------------------------------------#
    #	AGE
    #--------------------------------------------------------------#
    
    AGE_out = AGE(f4, out_channel//4)
    f4_out = Concatenate(axis=-1)([MSPP_out, AGE_out])
    
    # --------------------------------------------------------------------------------#
    #    f3_CIE
    # --------------------------------------------------------------------------------#
    
    f3_out = Conv2dT_BN(f4_out, out_channel//4, (3, 3))
    f3_CAT = SeparableConv2D(out_channel//4, kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(f3_CAT)
    f3_CAT = BatchNormalization()(f3_CAT)
    f3_CIE = CIE(f4_out, f3, f3_CAT, out_channel//4)
    f3_out = Concatenate(axis=-1)([f3_out, f3_CIE])
    f3_out = Conv2D(out_channel//4, (1,1), kernel_initializer = random_normal(stddev=0.02), padding='same', use_bias=False)(f3_out)
    f3_out = BatchNormalization()(f3_out)
    f3_out = Activation('relu')(f3_out)
    f3_out = DepthwiseConv2D(kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(f3_out)
    f3_out = BatchNormalization()(f3_out)   
    
    # --------------------------------------------------------------------------------#
    #    f2_CIE
    # --------------------------------------------------------------------------------#
    
    f2_out = Conv2dT_BN(f3_out, out_channel//8, (3, 3))
    f2_CAT = Conv2D(out_channel//8, kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(f2_CAT)
    f2_CAT = BatchNormalization()(f2_CAT)
    f2_CIE = CIE(f3_out, f2, f2_CAT, out_channel//8)
    f2_out = Concatenate(axis=-1)([f2_out, f2_CIE])
    f2_out = Conv2D(out_channel//8, (1,1), kernel_initializer = random_normal(stddev=0.02), padding='same', use_bias=False)(f2_out)
    f2_out = BatchNormalization()(f2_out)
    f2_out = Activation('relu')(f2_out)
    f2_out = DepthwiseConv2D(kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(f2_out)
    f2_out = BatchNormalization()(f2_out)
    
    # --------------------------------------------------------------------------------#
    #    f1_CIE
    # --------------------------------------------------------------------------------#    
    
    o = Conv2dT_BN(f2_out, out_channel//16, (3, 3))
    f1_out = Conv2D(out_channel//16, kernel_size=(3,3), strides = 1, activation='relu', padding='same', use_bias=False)(f1)
    f1_out = BatchNormalization()(f1_out)
    f1_CIE = CIE(f2_out, f1, f1_out, out_channel//16)
    f1_out = Concatenate(axis=-1)([o, f1_CIE])
    f1_out = Conv2D(out_channel//16, (1,1), kernel_initializer = random_normal(stddev=0.02), padding='same', use_bias=False)(f1_out)
    f1_out = BatchNormalization()(f1_out)
    f1_out = Activation('relu')(f1_out)
    f1_out = Conv2D(out_channel//16, (3,3), kernel_initializer = random_normal(stddev=0.02), padding='same', use_bias=False)(f1_out)
    f1_out = BatchNormalization()(f1_out)
    o = Activation('relu')(f1_out)
    
    o = Dropout(0.1)(o)
    o = Conv2D(num_classes,(1,1), kernel_initializer = random_normal(stddev=0.02), padding='same')(o)
    o = Lambda(resize_images)([o, img_input])
    o = Activation("softmax", name="main")(o)
    model = Model(img_input,[o])
    return model

if __name__ == "__main__": 
    model = fashionsegnet([480, 480, 3], 6)
    model.summary()
