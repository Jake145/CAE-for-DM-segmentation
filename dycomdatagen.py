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
def dycomrconverter(dir,end_path,dataframe,mask=False):
    a=time.perf_counter()
    print(f'time started:{a}')
    files=next(os.walk(dir))[2]

    #
    for f in files:


        if mask==True:
            dg = pydicom.dcmread(os.path.join(dir,f))
            good=dg.SeriesDescription
            if good=='ROI mask images':

                ds = pydicom.read_file(os.path.join(dir,f))
            elif good=='cropped images':
                pass
            else:
                print('error code not working')#replace with logging
        else:
            try:
                len(files)==1
            except:
                raise Exception(f'Nella directory {dir} delle singole immagini ci sono più file')
            ds = pydicom.read_file(os.path.join(dir, f))

    image = ds.pixel_array
    pattern=re.compile('P_0\d*')
    match=re.findall(pattern,dir)
    if 'MLO' in dir and 'LEFT' in dir:
        i=np.logical_and.reduce((df['patient_id']==match[0],df['image view']=='MLO',df['left or right breast']=='LEFT'))
    elif 'MLO' in dir and 'RIGHT' in dir:
        i=np.logical_and.reduce((df['patient_id']==match[0],df['image view']=='MLO',df['left or right breast']=='RIGHT'))
    elif 'CC' in dir and 'RIGHT' in dir:
        i=np.logical_and.reduce((df['patient_id']==match[0],df['image view']=='CC',df['left or right breast']=='RIGHT'))
    elif 'CC' in dir and 'LEFT' in dir:
        i=np.logical_and.reduce((df['patient_id']==match[0],df['image view']=='CC',df['left or right breast']=='LEFT'))
    else:
        raise Exception('Identificazione identificativo della immagine fallito')
    abn_type=dataframe['abnormality type'][i].array[0]
    pat_id=dataframe['patient_id'][i].array[0]
    pathology=dataframe['pathology'][i].array[0]
    view=dataframe['image view'][i].array[0]
    lor=dataframe['left or right breast'][i].array[0]
    if 'WindowWidth' in ds:
        print('Dataset has windowing')#rimpiazza con logging

    windowed = apply_voi_lut(image, ds)
    filename = f'{abn_type}_{pat_id}_{pathology}_{view}_{lor}.png'
    filename = os.path.join(end_path, filename)
    status = cv2.imwrite(filename,image)
    b=time.perf_counter()
    print(f'Time elapsed:{b-a}')
    return status

if __name__ == '__main__':

    ##define the datapaths
    datapath_mask_train='E:/massdata' #modificare con il proprio path
    datapath_mass_train='E:/massdatafull' #modificare con il proprio path
    main_directory='CBIS-DDSM'
    datapath_example=os.path.join(datapath_mass_train+main_directory)
    secondary_dir='Mass-Training_P_0'
    tertiary_dir='1.000000-ROI mask images'

    ##here we make the csv

    csv_path='C:/Users/pensa/Desktop/CAE/csvs'
    csv_filename='mass_case_description_train_set.csv'
    df = pd.read_csv(os.path.join(csv_path,csv_filename))


    ##now we create the png files for the full image

    dirpath=os.path.join(datapath_mass_train,main_directory,secondary_dir+'*','*','*')
    directories=glob.glob(dirpath)
    endpath='E:/Mass_data'
    if not os.path.exists(endpath):
        os.makedirs(endpath)
    trpath='Train_data'
    endpath_1=os.path.join(endpath,trpath)
    if not os.path.exists(endpath_1):
        os.makedirs(endpath_1)

    ## Definiamo la funzione da mappare nell'esecutore del MT


    dycomrconverter=partial(dycomrconverter,end_path=endpath_1,dataframe=df,mask=False)

    ##multi thread
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(dycomrconverter, directories)

    ##Definiamo i path per le maschere

    dirpath_mask=os.path.join(datapath_mask_train,main_directory,secondary_dir+'*','*','*')
    directories_mask=glob.glob(dirpath_mask)
    trpath_mask='Train_data_masks'
    endpath_2=os.path.join(endpath,trpath_mask)
    if not os.path.exists(endpath_2):
        os.makedirs(endpath_2)

    ## Definiamo la funzione da mappare nell'esecutore del MT


    dycomrconverter_masks=partial(dycomrconverter,end_path=endpath_2,dataframe=df,mask=True)

    ##multi thread
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = executor.map(dycomrconverter_masks, directories_mask)

    end=time.perf_counter()

    print(f'Elapsed time for MT:{end-start}')




