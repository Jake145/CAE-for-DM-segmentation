import numpy as np
import os
from PIL import Image
from skimage.io import imread
import datetime
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from keras.constraints import unit_norm,min_max_norm,max_norm
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import glob
import radiomics
from radiomics import featureextractor
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
#import sys
#sys.path
#sys.path.append('C:/Users/pensa/Desktop/CAE-for-DM-segmentation/functioncae')
import time
LOG_DIR = f"{int(time.time())}"
from functioncae import caehelper,ClassesCAE
import kerastuner
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
import time
LOG_DIR = f"{int(time.time())}"

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Tuner.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

from keras.layers import Conv2D, Conv2DTranspose, Input, Dropout,MaxPooling2D, UpSampling2D, Dense, Flatten
from keras.models import Model, load_model
from keras.layers.experimental.preprocessing import Resizing
from keras.layers.merge import concatenate

def build_model_rad_UNET(hp,shape=(124,124,1),feature_dim=(3,)):
    input_tensor = Input(shape=shape)
    input_vector= Input(shape=feature_dim)

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
    den = Dense(hp.Int(f'dense_base_unit',
                                min_value=32,
                                max_value=64,
                                step=4), activation='relu')(flat)

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.


      den = Dense(hp.Int(f'conv_{i}_units',
                                min_value=4,
                                max_value=64,
                                step=4), activation='relu')(den)
      den= Dropout(hp.Float(f'drop_{i}_rate',
                                min_value=0,
                                max_value=.5,
                                step=.1,))(den)

    classification_output = Dense(2, activation = 'sigmoid', name="classification_output")(den)

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
    u8 = Resizing(62,62,interpolation='nearest')(c2)

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
    model.compile(optimizer='adam', loss={'decoder_output':'binary_crossentropy','classification_output':'categorical_crossentropy'},
                  metrics={'decoder_output':'MAE','classification_output':tf.keras.metrics.AUC()})
    return model

datapath='C:/Users/pensa/Desktop/CAE-for-DM-segmentation/large_sample_Im_segmented_ref'

X_rad,Y_rad,LABELS_rad = caehelper.read_dataset(datapath,'pgm','_2_resized','_1_resized')
X_rad = X_rad/255
Y_rad = Y_rad/255


"""#In questa sezione estraiamo le feature con pyradiomics e facciamo la PCA

Ora si estraggono le feature e si mettono in un dizionario che verrà poi convertito in un pandas dataframe
"""

x_id ="_resized"
y_id="_mass_mask"
ext='pgm'
fnames = glob.glob(os.path.join(datapath, f"*{x_id}.{ext}"))
fnamesmask = glob.glob(os.path.join(datapath, f"*{y_id}.{ext}"))

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('shape2D')
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')

dataframe={f.replace(datapath,''):extractor.execute(caehelper.read_pgm_as_sitk(f), caehelper.read_pgm_as_sitk(f.replace(x_id,y_id)),label=255) for f in fnames}


Pandata=pd.DataFrame(dataframe)

Pandataframe=Pandata.drop(Pandata.index[0:22]).T


X_train_rad, X_test_rad, Y_train_rad, Y_test_rad,class_train_rad,class_test_rad,feature_train,feature_test = train_test_split(X_rad, Y_rad,LABELS_rad,Pandataframe, test_size=0.2, random_state=42)



sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)


pca = PCA()
feature_train = pca.fit_transform(feature_train)
feature_test = pca.transform(feature_test)
explained_variance = pca.explained_variance_ratio_


pca = PCA(n_components=3)
feature_train = pca.fit_transform(feature_train)
feature_test = pca.transform(feature_test)



train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip= True,
        fill_mode='reflect')

transform = train_datagen.get_random_transform((124,124))


X_train_rad_tr, X_train_rad_val, Y_train_rad_tr, Y_train_rad_val,class_train_rad_tr,class_train_rad_val,feature_train_tr,feature_train_val = train_test_split(X_train_rad, Y_train_rad, to_categorical(class_train_rad,2),feature_train, test_size=0.2, random_state=24)

mass_gen_rad = ClassesCAE.MassesSequence_radiomics(X_train_rad_tr, Y_train_rad_tr,class_train_rad_tr,feature_train_tr, train_datagen)

tuner = BayesianOptimization(
    build_model_rad_UNET,
    objective= kerastuner.Objective("val_classification_output_auc", direction="max"),
    max_trials=20,
    executions_per_trial=1,
    directory=LOG_DIR)

tuner.search(mass_gen_rad,
             verbose=2,
             epochs=100,
             batch_size=len(mass_gen_rad),
             #callbacks=[tensorboard],
             validation_data=([X_train_rad_val,feature_train_val], [Y_train_rad_val,class_train_rad_val]))

tuner.get_best_hyperparameters()[0].values

newmod=build_modelUNET(tuner.get_best_hyperparameters()[0])
newmod.summary()

tuner.get_best_models()[0].summary()



newmod.save('model_Unet_Tuned') #the model_opt is better

