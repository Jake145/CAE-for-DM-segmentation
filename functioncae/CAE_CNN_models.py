import tensorflow as tf
import datetime, os

from keras.layers import Conv2D, Conv2DTranspose, Input, Dropout,MaxPooling2D, UpSampling2D, Dense, Flatten
from keras.models import Model, load_model
from keras.layers.experimental.preprocessing import Resizing
from keras.layers.merge import concatenate
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Models.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
def make_model_rad_REGULIZER(shape_tensor=(124,124,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    input_vector= Input(shape=feature_dim)
    logger.debug(f'dimensione input feature:{feature_dim}')

    x = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(input_tensor)
    x = Dropout(.2,)(x)
    x = MaxPooling2D((2, 2), strides=(2,2),padding='same')(x)
    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Dropout(.2,)(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu',name='last_conv')(x)

    flat=Flatten()(x)
    flat=concatenate([flat,input_vector])
    den = Dense(16, activation='relu')(flat)
    #den= Dropout(.1,)(den)




    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(den)

    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Dropout(.2,)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    decoder_out = Conv2D(1, (5,5), padding='valid',activation='sigmoid',name="decoder_output")(x)
    model = Model([input_tensor,input_vector], [decoder_out,classification_output])

    return model

def make_model_rad(shape_tensor=(124,124,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    input_vector= Input(shape=feature_dim)
    logger.debug(f'dimensione input feature:{feature_dim}')

    x = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(input_tensor)

    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)

    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu',name='last_conv')(x)

    flat=Flatten()(x)
    flat=concatenate([flat,input_vector])
    den = Dense(16, activation='relu')(flat)


    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(flat)

    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    decoder_out = Conv2D(1, (5,5), padding='valid',activation='sigmoid',name="decoder_output")(x)
    model = Model([input_tensor,input_vector], [decoder_out,classification_output])

    return model

from tensorflow.keras import regularizers
##

def make_model_rad_UNET(shape_tensor=(124,124,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    input_vector= Input(shape=feature_dim)
    logger.debug(f'dimensione input feature:{feature_dim}')

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal',
                            padding='same')(c1)
    p1 =MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Resizing(32,32,interpolation='nearest')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p2)


    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Resizing(16,16,interpolation='nearest')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same', name="last_conv")(c5)
#fc layers

    flat=Flatten()(c5)
    flat=concatenate([flat,input_vector])
    den = Dense(16, activation='relu')(flat)


    classification_output = Dense(2, activation = 'softmax', name="classification_output")(flat)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Resizing(62,62,interpolation='nearest')(c2)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c9)


    decoder_out = Conv2D(1, (1, 1), activation='sigmoid',name="decoder_output")(c9)

    model = Model([input_tensor,input_vector], [decoder_out,classification_output])
    return model


def make_model(shape_tensor=(124,124,1)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')


    x = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(input_tensor)
    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu',name='last_conv')(x)

    flat=Flatten()(x)
    den = Dense(16, activation='relu')(flat)
    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(flat)

    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    decoder_out = Conv2D(1, (5,5), padding='valid',activation='sigmoid',name="decoder_output")(x)
    model = Model(input_tensor, [decoder_out,classification_output])

    return model

"""Model 2 is the same but with added regularization (dropout layers) and maxpooling


"""

def make_modelREGULIZER(shape_tensor=(124,124,1)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    x = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(input_tensor)
    x = Dropout(.2,)(x)
    x = MaxPooling2D((2, 2), strides=(2,2),padding='same')(x)
    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Dropout(.2,)(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu',name='last_conv')(x)

    flat=Flatten()(x)
    den = Dense(16, activation='relu')(flat)
    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(flat)


    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Dropout(.2,)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    decoder_out = Conv2D(1, (5,5), padding='valid',activation='sigmoid',name="decoder_output")(x)
    model = Model(input_tensor, [decoder_out,classification_output])

    return model

"""This model is the Unet from Ronneberger e al, U-Net: Convolutional Networks for Biomedical
Image Segmentation. I added a resizing layer to adapt it for our image size

"""

from keras.constraints import unit_norm,min_max_norm,max_norm
from tensorflow.keras import regularizers


def make_modelUNET(shape_tensor=(124,124,1)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal',
                            padding='same')(c1)
    p1 =MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Resizing(32,32,interpolation='nearest')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p2)


    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same', name="last_conv")(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Resizing(16,16,interpolation='nearest')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c5)
#fc layers

    flat=Flatten()(c3)
    den = Dense(16, activation='relu')(flat)
    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(flat)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Resizing(62,62,interpolation='nearest')(c2)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                            padding='same')(c9)


    decoder_out = Conv2D(1, (1, 1), activation='sigmoid',name="decoder_output")(c9)

    model = Model(input_tensor, [decoder_out,classification_output])
    return model

def make_model_rad_BIG_REGULIZER(shape_tensor=(4096,3072,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    input_vector= Input(shape=feature_dim)
    logger.debug(f'dimensione input feature:{feature_dim}')
    x = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(input_tensor)
    x = Dropout(.2,)(x)
    x = MaxPooling2D((2, 2), strides=(2,2),padding='same')(x)
    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Dropout(.2,)(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu',name='last_conv')(x)

    flat=Flatten()(x)
    flat=concatenate([flat,input_vector])
    den = Dense(16, activation='relu')(flat)
    #den= Dropout(.1,)(den)




    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(den)

    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Dropout(.2,)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    decoder_out = Conv2D(1, (5,5), padding='valid',activation='sigmoid',name="decoder_output")(x)
    model = Model([input_tensor,input_vector], [decoder_out,classification_output])
        return model

def make_model_rad_BIG(shape_tensor=(4096,3072,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    input_vector= Input(shape=feature_dim)
    logger.debug(f'dimensione input feature:{feature_dim}')

    x = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(input_tensor)
    #x = Dropout(.2)(x)
    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    #x = Dropout(.2)(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu',name='last_conv')(x)

    flat=Flatten()(x)
    flat=concatenate([flat,input_vector])
    den = Dense(16, activation='relu')(flat)
    #den = Dropout(.2)(den)

    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(flat)

    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    decoder_out = Conv2D(1,(1,1) , padding='valid',activation='sigmoid',name="decoder_output")(x)
    model = Model([input_tensor,input_vector], [decoder_out,classification_output])

    return model

def make_model_rad_BIG_UNET(shape_tensor=(4096,3072,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    logger.debug(f'dimensione input immagine:{shape_tensor}')
    input_vector= Input(shape=feature_dim)
    logger.debug(f'dimensione input feature:{feature_dim}')

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(input_tensor)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal',
                                padding='same')(c1)
    p1 =MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
        #p2 = Resizing(32,32,interpolation='nearest')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p2)


    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same', name="last_conv")(c3)
    p3 = MaxPooling2D((2, 2))(c3)
        #p3 = Resizing(16,16,interpolation='nearest')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c4)

    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(p4)

    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c5)
    #fc layers

    flat=Flatten()(c3)
    flat=concatenate([flat,input_vector])
    den = Dense(16, activation='relu')(flat)


    classification_output = Dense(2, activation = 'softmax', name="classification_output")(flat)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

        #c4 = Resizing(14,14,interpolation='nearest')(c4)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        #c3= Resizing(28,28,interpolation='nearest')(c3)

    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        #u8 = Resizing(62,62,interpolation='nearest')(c2)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        #c1= Resizing(112,112,interpolation='nearest')(c1)

    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',
                                padding='same')(c9)


    decoder_out = Conv2D(1, (1, 1), activation='sigmoid',name="decoder_output")(c9)

    model = Model([input_tensor,input_vector], [decoder_out,classification_output])
    return model