import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import pickle
import SimpleITK as sitk
import radiomics
import re
import time
import pickle
import concurrent.futures
from functools import partial
import radiomics
from radiomics import featureextractor
sys.path.append('C:/Users/pensa/Desktop/CAE-for-DM-segmentation/functioncae')
from caehelper import *
import warnings



def resizer(list,endpath,pattern):

    a=time.perf_counter()
    try:
        image=Image.open(list[0])
    except:
        warnings.warn(f'Immagine {list[0]} mancante o corrotta')

    try:
        mask=Image.open(list[1])
    except:
        warnings.warn(f'Immagine {list[0]} mancante o corrotta')

    try:

        mask = mask.resize((image.shape))
        b=time.perf_counter()
        match=re.findall(pattern,list[0])[0]
        filename=os.path.join(endpath,match+'.png')
        mask.save(filename)
    except:
        warnings.warn(f'Non possibile andare avanti')


    return f'time elapsed: {b-a}'


##
if __name__ == '__main__':

    datapath='E:'
    maindir='Mass_data_new'
    mass_train='Train_data'
    mass_test='Test_data'
    mask_train='Train_data_masks'
    mask_test='Test_data_masks'
    mask_train_res='resized_masks'
    benign_label='BENIGN'
    malign_label='MALIGNANT'
    features_path='feats'

    path_mass_tr=os.path.join(datapath,maindir,mass_train)
    path_masks_tr=os.path.join(datapath,maindir,mask_train)

    path_masks_resized=os.path.join(datapath,maindir,mask_train_res)


    X_big_train,Y_big_train,Class_big_train=read_dataset_big(path_mass_tr
    ,path_masks_tr,benign_label,malign_label)


    if not os.path.exists(path_masks_resized):
        os.makedirs(path_masks_resized)


    ##
    """#Pyradiomics on big dataset"""

    endpath_tr=os.path.join(datapath,maindir,features_path)
    if not os.path.exists(endpath_tr):
        os.makedirs(endpath_tr)
##


    """Questa funzione serve per estrarre le feature in multiprocessing e aggiungerle a un dizionario"""





    extractor = featureextractor.RadiomicsFeatureExtractor()


    biggy=[[X_big_train[i],Y_big_train[i]] for i in range(len(X_big_train))]
    #biggy_test=[[X_big_test[i],Y_big_test[i]] for i in range(len(X_big_test))]

    #
    ##
    rez=partial(resizer,endpath=path_masks_resized,pattern=re.compile(r'[M][\w-]*[0-9]*[\w]{13}'))
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(rez, biggy)
        print(results)
    end=time.perf_counter()

    print(f'Elapsed time for MT:{end-start}')

##

    X_big_train_n,Y_big_train_n,Class_big_train_n=read_dataset_big(path_mass_tr
    ,path_masks_resized,benign_label,malign_label)

    biggy=[[X_big_train_n[i],Y_big_train_n[i]] for i in range(len(X_big_train_n))]

##

    nam=radiomic_dooer(biggy[-1],path_mass_tr,endpath_tr,255,extractor)
    radiomic_dooer_new=partial(radiomic_dooer,datapath=path_mass_tr,endpath=endpath_tr,label=255,extrc=extractor)
    ##
    #this is the filename list for the multiprocessing
    errors=[]
    for item in biggy:
        try:
            name=radiomic_dooer(item,path_mass_tr,endpath_tr,255,extractor)
        except:
            errors.append(item)
##
