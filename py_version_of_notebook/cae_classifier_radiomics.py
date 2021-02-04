
##

import numpy as np
import os
import pickle
import shutil
import cv2
import os
import numpy as np
import glob
from skimage.io import imread
import time
import glob
import matplotlib.pyplot as plt
from skimage import io, transform
from keras.utils import to_categorical
import PIL
from sklearn.model_selection import train_test_split
import sys
sys.path
sys.path.append('C:/Users/pensa/Desktop/CAE-for-DM-segmentation/functioncae')
from caehelper import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('RadiomicsSegm.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
##




datapath='C:/Users/pensa/Desktop/CAE-for-DM-segmentation/large_sample_Im_segmented_ref'

X_rad,Y_rad,LABELS_rad = read_dataset(datapath,'pgm','_2_resized','_1_resized')
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

dataframe={f.replace(datapath,''):extractor.execute(read_pgm_as_sitk(f), read_pgm_as_sitk(f.replace(x_id,y_id)),label=255) for f in fnames}

import pandas as pd

Pandata=pd.DataFrame(dataframe)



"""Qui vediamo tutti i campi. Naturalmente quelli che riguardano la versione dei pacchetti non interessano, quindi li eliminiamo"""

#Pandata.index logging

"""Con questo ciclo si trovano i campi che non interessano"""

for i,name in enumerate(Pandata.index):
  if 'diagnostics' in Pandata.index[i]:
    print(i)  
  else:
    pass

"""Eliminiamo quei campi e facciamo la trasposta"""

Pandataframe=Pandata.drop(Pandata.index[0:22]).T



"""Ora che abbiamo le feature, facciamo una pca per trovare le componenti che descrivono al meglio la variazione dei dati"""

X_train_rad, X_test_rad, Y_train_rad, Y_test_rad,class_train_rad,class_test_rad,feature_train,feature_test = train_test_split(X_rad, Y_rad,LABELS_rad,Pandataframe, test_size=0.2, random_state=42)

"""Per prima cosa si fa la rinormalizzazione dei dati"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)

from sklearn.decomposition import PCA

pca = PCA()
feature_train = pca.fit_transform(feature_train)
feature_test = pca.transform(feature_test)
explained_variance = pca.explained_variance_ratio_

#explained_variance logging

"""Abbiamo trovato 93 feature, ma come si vede solo le prime sono significative. Guardiamo quali sono quelle che descrivono maggiormente la variazione"""

percentage_var_explained = pca.explained_variance_ratio_;  
cum_var_explained=np.cumsum(percentage_var_explained)
#plot spettro della PCA    
plt.figure(figsize=(6,4))
plt.clf()  
plt.plot(cum_var_explained,linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('n_components') 
plt.ylabel('Cumulative_Variance_explained')  
plt.show()
plt.close()

"""Stampiamo quanta variazione si descrive al variare del numero di componenti"""

exp_var_cumsum=pd.Series(np.round(pca.explained_variance_ratio_.cumsum(),4)*100)  
for index,var in enumerate(exp_var_cumsum):  
    print('if n_components= %d,   variance=%f' %(index,np.round(var,3)))

"""Manteniamo le prime tre"""

from sklearn.decomposition import PCA

pca = PCA(n_components=3) 
feature_train = pca.fit_transform(feature_train)
feature_test = pca.transform(feature_test)

"""#Ora si prosegue importando i modelli con in input anche le features e si definisce una nuova classe di generatore che permette di avere anche le feature"""

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

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
#transform

import keras
class MassesSequence_radiomics(keras.utils.Sequence):
    """ Classe per il data augmentation per CAE con feature radiomiche """

    def __init__(self, x, y,label_array,features, img_gen, batch_size=10, shape=(124,124)):
        """ Inizializza la sequenza

        Parametri::

        x (np.array): immagini
        y (np.array): maschere
        label_array (np.array): label di classificazione
        features (np.array): feature ottenute con pyradiomics
        batch_size (int): dimensioni della batch
        img_gen (ImageDatagenerator): istanza di ImageDatagenerator
        shape (tuple): shape dell'immagine. Di Default (124, 124)

        """
        self.x, self.y,self.label_array,self.features = x, y,label_array,features
        self.shape = shape
        self.img_gen = img_gen
        self.batch_size = batch_size


    def __len__(self):
        return len(self.x) // self.batch_size

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        self.x, self.y ,self.label_array,self.features= shuffle(self.x, self.y,
                                                                self.label_array,self.features)

    def process(self, img, transform):
        """ Apply a transformation to an image """
        img = self.img_gen.apply_transform(img, transform)
        return img
            
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_array = self.label_array[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_features = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]


        X=[];
        Y=[];
        Classes=[];
        Features=[]
        
        for image, mask,label,feature in zip(self.x, self.y,self.label_array,self.features):
            transform = self.img_gen.get_random_transform(self.shape)
            X.append(self.process(image, transform))
            Y.append(self.process(mask, transform)>0.2)
            Classes.append(label)
            Features.append(feature)

          
        return [np.asarray(X,np.float64),np.asarray(Features,np.float64)], [np.asarray(Y,np.float64) ,np.asarray(Classes,np.float)]

X_train_rad_tr, X_train_rad_val, Y_train_rad_tr, Y_train_rad_val,class_train_rad_tr,class_train_rad_val,feature_train_tr,feature_train_val = train_test_split(X_train_rad, Y_train_rad, to_categorical(class_train_rad,2),feature_train, test_size=0.2, random_state=24)

mass_gen_rad = MassesSequence_radiomics(X_train_rad_tr, Y_train_rad_tr,class_train_rad_tr,feature_train_tr, train_datagen)

batch=mass_gen_rad[6]

#batch[0][0].shape[1:] logging

"""#Definiamo i modelli"""



import tensorflow as tf
import datetime, os

from keras.layers import Conv2D, Conv2DTranspose, Input, Dropout,MaxPooling2D, UpSampling2D, Dense, Flatten
from keras.models import Model, load_model
from keras.layers.experimental.preprocessing import Resizing
from keras.layers.merge import concatenate

def make_model_rad_REGULIZER(shape_tensor=batch[0][0].shape[1:],feature_dim=batch[0][1].shape[1:]):
    input_tensor = Input(shape=shape_tensor,name="tensor_input")
    input_vector= Input(shape=feature_dim)

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

def make_model_rad(shape_tensor=batch[0][0].shape[1:],feature_dim=batch[0][1].shape[1:]):
    input_tensor = Input(shape=shape_tensor)
    input_vector= Input(shape=feature_dim)
    
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
 
def make_model_rad_UNET(shape_tensor=batch[0][0].shape[1:],feature_dim=batch[0][1].shape[1:]):
    input_tensor = Input(shape=shape_tensor)
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
##
"""Ora importiamo i modelli e facciamo il training. Questa parte del notebook contiene parti di codice simile alla versione senza pyradiomics, dunque non vi sono riportati i commenti di quella versione"""



"""importiamo i modelli con input sia l'immagine che le feature"""

import tensorflow as tf
import datetime, os
##
model_rad = make_model_rad_UNET() 
model_rad.summary()

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

checkpoint_filepath = 'C:/Users/pensa/Desktop/CAE-for-DM-segmentation/models/rad_Unet_weights.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_classification_output_auc',
    mode='max',
    save_best_only=True)

model_rad.compile(optimizer='adam', loss={'decoder_output':'binary_crossentropy','classification_output':'categorical_crossentropy'},
                  metrics={'decoder_output':'MAE','classification_output':tf.keras.metrics.AUC()})

epoch_number= 200

history_rad = model_rad.fit(mass_gen_rad, steps_per_epoch=len(mass_gen_rad), epochs=epoch_number, 
                        validation_data=([X_train_rad_val,feature_train_val] ,[Y_train_rad_val,class_train_rad_val]),
                        callbacks=[model_checkpoint_callback])
##
modelviewer(history_rad)
##

"""Carichiamo il modello con le prestazioni migliori"""

model_rad = keras.models.load_model('C:/Users/pensa/Desktop/CAE-for-DM-segmentation/models/rad_Unet_weights.17-0.56.h5')

"""Visualizziamo alcune immagini ottenute con la rete"""

idx=67
xtrain = X_train_rad[idx][np.newaxis,...]
ytrain = Y_train_rad[idx][np.newaxis,...]
#xtrain.shape logging

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.imshow(xtrain.squeeze())
plt.subplot(1,3,2)
plt.imshow(ytrain.squeeze())
plt.subplot(1,3,3)
plt.imshow(otsu(model_rad.predict([xtrain,feature_train[idx][np.newaxis,...]])[0].squeeze()))

"""Ora su immagini di test"""

idx=16
xtest = X_test_rad[idx][np.newaxis,...]
ytest = Y_test_rad[idx][np.newaxis,...]

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.imshow(xtest.squeeze())
plt.subplot(1,3,2)
plt.imshow(ytest.squeeze())
plt.subplot(1,3,3)
plt.imshow(otsu(model_rad.predict([xtest,feature_test[idx][np.newaxis,...]])[0].squeeze()))

"""Valutiamo i coefficienti di dice medi

The average Dice on the train set is:
"""

dicetr=dice_vectorized(Y_train_rad,otsu(model_rad.predict([X_train_rad,feature_train])[0])).mean()

docetest=dice_vectorized(Y_test_rad,otsu(model_rad.predict([X_test_rad,feature_test])[0])).mean()

"""Visualizziamo una heatmap di attivazione del layer convoluzionale prima del fully connected del classificatore"""

hmap=heatmap_rad(X_test_rad[18],feature_test[18],model_rad)

"""Vediamo ora la curva roc e l'AUC"""
##
from sklearn.metrics import roc_curve
y_pred = model_rad.predict([X_test_rad,feature_test])[1]
fpr, tpr, thresholds = roc_curve(class_test_rad, [item[0] for _,item in enumerate(y_pred)],pos_label=0)

from sklearn.metrics import auc
auc = auc(fpr, tpr)

plot_roc_curve(fpr, tpr,auc)