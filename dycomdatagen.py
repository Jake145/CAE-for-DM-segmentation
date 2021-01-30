import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from skimage import data, exposure, img_as_float
import pandas as pd
import cv2
import shutil
import concurrent.futures
import time
import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from skimage import data, exposure, img_as_float
import pandas as pd
import cv2
import shutil
import concurrent.futures
import time
from functools import partial
import re

import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Dicomdatagen.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def dycomrconverterV2(data,dir_mass,dir_mask,end_path_mass,end_path_mask):
    logger.debug(f'Per estrarre le immagini e maschere, carico da {dir_mass} e {dir_mask} con cartelle di arrivo {end_path_mass} e {end_path_mask}')
    def search_files(directory='.', extension=''):
        extension = extension.lower()
        filelist=[]
        for dirpath, dirnames, files in os.walk(directory):
            for name in files:
                if extension and name.lower().endswith(extension):
                    filelist.append(os.path.join(dirpath, name))
                elif not extension:
                    filelist.append(os.path.join(dirpath, name))
        return filelist
    a=time.perf_counter()
    logger.info(f'time started:{a}')
    pattern=re.compile(r'[\w-]+[\w]\b')
    match_mass=re.findall(pattern,data['image file path'])[0] #data=df.iloc[i,:]
    logger.debug(f'trovato come nome della immagine: {match_mass}')
    match_mask=re.findall(pattern,data['ROI mask file path'])[0] #data=df.iloc[i,:]
    logger.debug(f'trovato come nome della maschera: {match_mask}')


    dir_masses=os.path.join(dir_mass,match_mass)
    logger.info(f'impostato come directory di arrivo per le immagini: {dir_masses}')


    files=search_files(dir_masses,'.dcm')
    logger.info('sto cercando i dycom in {dir_masses}')
    try:
        len(files)==1
        logger.debug('ho verificato che nella cartella vi è una sola immagine')
        #questo va bene nel nostro caso specifico. Nel caso generale sarebbe meglio guardare il nome del file dal dycom, ma questo modo è più rapido
    except:
        raise Exception(f'Nella directory {dir_masses} delle singole immagini ci sono più file')
        logger.exception(f'Nella directory {dir_masses} delle singole immagini ci sono più file')



    f=files[0]
    logger.info(f'ho letto {f}')
    ds = pydicom.read_file(f)
    logger.info(f'ho letto correttamente il dicom di {f}')
    image = ds.pixel_array

    pathology=data['pathology']

    if 'WindowWidth' in ds:
        logger.info('Dataset ha windowing')

    windowed = apply_voi_lut(image, ds)
    filename = f'{match_mass}_{pathology}.png'
    filename = os.path.join(end_path_mass, filename)
    logger.debug(f'salvato in {filename}')
    status_mass = cv2.imwrite(filename,image)
    logger.info(f'correttamente salvato {filename}')

    #ora facciamo la maschera
    dir_masks=os.path.join(dir_mask,match_mask)
    logger.info(f'impostato come directory di arrivo per le maschere: {dir_masks}')
    files=search_files(dir_masks,'.dcm')
    logger.info('sto cercando i dycom in {dir_masks}')

    try:
        len(files)<=2
        logger.debug('ho verificato che nella cartella vi è al più due immagini')

    except:
        raise Exception(f'Nella directory {dir_masks} delle maschere ci sono più masks')
        logger.exception(f'Nella directory {dir_masks} delle maschere ci sono più di due file')
    for f in files:

        dg = pydicom.dcmread(f)
        logger.debug(f'ho letto correttamente il dicom di {f}')
        good=dg.SeriesDescription
        if good=='ROI mask images':

            ds_mask = pydicom.read_file(f)
            logger.debug(f'in {f} vi è una maschera')
        elif good=='cropped images':
            logger.debug(f'in {f} vi è una ROI')

            pass
        else:
            raise Exception(f'{f} File sconosciuto!')
            logger.warning(f'{f} File sconosciuto!')

    image_mask = ds_mask.pixel_array





    if 'WindowWidth' in ds_mask:
        logger.info('Dataset ha windowing')

    windowed = apply_voi_lut(image_mask, ds_mask)
    filename_mask = f'{match_mass}_{pathology}.png'
    filename_mask = os.path.join(end_path_mask, filename_mask)
    logger.debug(f'salvato in {filename}')

    status_mask = cv2.imwrite(filename_mask,image_mask)
    logger.info(f'maschera correttamente salvato {filename}')

    b=time.perf_counter()
    logger.info(f'Time elapsed:{b-a}')
    return status_mass,status_mask



###

if __name__ == '__main__':



#Definiamo i path e il dataframe


    datapath_mask_train='E:/massdata' #modificare con il proprio path
    datapath_mass_train='E:/massdatafull' #modificare con il proprio path
    main_directory='CBIS-DDSM'



    csv_path='C:/Users/pensa/Desktop/CAE/csvs'
    csv_filename='mass_case_description_train_set.csv'
    df = pd.read_csv(os.path.join(csv_path,csv_filename))

    datas=[]
    for i in range(len(df)):
        datas.append(df.iloc[i,:])

    directory_mass=os.path.join(datapath_mass_train,main_directory)
    directory_mask=os.path.join(datapath_mask_train,main_directory)


    endpath='E:/Mass_data_new'
    if not os.path.exists(endpath):
        os.makedirs(endpath)
    logger.info(f'creato il path {endpath}')


    trpath='Train_data'
    endpath_1=os.path.join(endpath,trpath)
    if not os.path.exists(endpath_1):
        os.makedirs(endpath_1)
    logger.info(f'creato il path {endpath_1}')

    trpath_mask='Train_data_masks'
    endpath_2=os.path.join(endpath,trpath_mask)
    if not os.path.exists(endpath_2):
        os.makedirs(endpath_2)
    logger.info(f'creato il path {endpath_2}')



    dycomrconverter_new=partial(dycomrconverterV2,dir_mass=directory_mass,dir_mask=directory_mask,end_path_mass=endpath_1,end_path_mask=endpath_2)


    ##multi thread
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(dycomrconverter_new, datas)
        print(results)
    end=time.perf_counter()

    logger.info(f'Elapsed time for MT:{end-start}')







