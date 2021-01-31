import PIL
from PIL import Image
import cv2
import os
import numpy as np
import glob
from skimage.io import imread
import time
import re
import warnings
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('CAE_functions.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def save_newext(file_name,data_path,ext1,ext2,endpath):
  """
  Riscrive le immagini in formato leggibile per pyradiomics
  type file_name: stringa
  param file_path: nome del file della immagine

  type data_path: stringa
  param data_path: percorso della cartella dove si trova la immagine

  type ext1: stringa
  param ext1: stringa identificativa dell'estenzione di partenza della immagine

  type ext2: stringa
  param ext2: stringa identificativa dell'estenzione finale della immagine

  type endpath: stringa
  param endpath: percorso della cartella di arrivo
  """
  if ext1==ext2:
    logger.debug(f'il file {file_name} in {data_path} ha già la estenzione {ext2}' )
  try:
    image=plt.imread(os.path.join(data_path,file_name))
    file_name=file_name.replace(f'.{ext1}',f'.{ext2}') #insert logging warning if ext1==ext2
    logger.info(f'read {os.path.join(data_path,file_name)} and changed extension from {ext1} to {ext2}')
  except:
    raise Exception('immagine o path non trovati')
    logger.exception(f'Non ho trovato {file_name} in {data_path}')
  status = cv2.imwrite(os.path.join(endpath,file_name),image)
  logger.info(f'ho scritto il file {file_name} in {endpath} come .{ext2} ')
  return status

def unit_masks(file_name,data_path,ext1,ext2, endpath):
  """
  Normalizza i valori dei pixel delle maschere già nei file per essere utilizzati con pyradiomics.
  Permette inoltre di cambiare l'estenzione da .pgm a .png o qualunque altra estenzione supportata.

  type file_name: stringa
  param file_path: nome del file della maschera

  type data_path: stringa
  param data_path: percorso della cartella dove si trova la maschera

  type ext1: stringa
  param ext1: stringa identificativa dell'estenzione di partenza della maschera

  type ext2: stringa
  param ext2: stringa identificativa dell'estenzione finale della maschera

  type endpath: stringa
  param endpath: percorso della cartella di arrivo
  """
  try:
    image=plt.imread(os.path.join(data_path,file_name))
    logger.info(f'Ho letto {file_name} in {data_path}')
  except:
    raise Exception('immagine o path non trovati!')
    logger.exception(f'{file_name} non trovato in {data_path}')
  image=image/255
  file_name=file_name.replace(f'.{ext1}',f'.{ext2}')
  status = cv2.imwrite(os.path.join(endpath,file_name),image)
  logging.info(f'ho scritto {file_name} in {endpath} con successo')
  return status,image

def read_dataset(dataset_path,ext,benign_label,malign_label,x_id ="_resized", y_id="_mass_mask"):
  """
  Data la cartella con le maschere e le immagini, restituisce i vettori con le immagini, le maschere e le classi.
  Restituisce i vettori come tensori da dare alla rete.

  type dataset_path: stringa
  param dataset_path: Cartella con le immagini e le relative maschere

  type data_path: stringa
  param data_path: percorso della cartella dove si trova la maschera

  type ext: stringa
  param ext: stringa identificativa dell'estenzione delle immagini e maschere

  type x_id: stringa
  param x_id: identificativo delle immagini

  type x_id: stringa
  param x_id: identificativo delle maschere

  type benign_label: stringa
  param benign_label: identificativo delle masse benigne

  type malign_label: stringa
  param malign_label: identificativo delle masse maligne

  """

  fnames = glob.glob(os.path.join(dataset_path, f"*{x_id}.{ext}"))
  logger.info(f'ho analizzato {dataset_path} cercando le immagini')
  if fnames == []:
    raise Exception('Niente immagini! Il path è sbagliato, magari x_id o ext sono sbagliati! ')
    logger.exception('Non ho trovato immagini in {dataset_path}')
  else:
    pass
  X = []
  Y = []
  class_labels=[]
  for fname in fnames:
      X.append(plt.imread(fname)[1:,1:,np.newaxis])
      Y.append(plt.imread(fname.replace(x_id, y_id))[1:,1:,np.newaxis])
      if benign_label in fname:
        class_labels.append(0)
      elif malign_label in fname:
        class_labels.append(1)
  logger.info(f'ho preso le immagini e le maschere da {dataset_path}, ho trovato tutto e ho creato gli array delle immagini, maschere e classi')
  return np.array(X), np.array(Y) , np.array(class_labels)

def read_dataset_big(dataset_path_mass,dataset_path_mask,benign_label,malign_label,ext='png'):
  """
  Versione di read_dataset per il dataset del TCIA.
  Data la cartella con le maschere e le immagini, restituisce i vettori con i filepath delle immagini, le maschere e le classi.
  Restituisce i vettori come tensori da dare al generatore per la rete.

  type dataset_path_mass: stringa
  param dataset_path_mass: Cartella con le immagini

  type dataset_path_mask: stringa
  param dataset_path_mask: percorso della cartella dove si trovano le maschera

  type ext: stringa
  param ext: stringa identificativa dell'estenzione delle immagini e maschere

  type resize: bool
  param resize: Se TRUE fa il reshape delle maschere per combaciare con quello delle immagini



  """
  fnames  = glob.glob(os.path.join(dataset_path_mass, f"*.{ext}"))
  logger.info(f'ho analizzato {dataset_path_mass} cercando le immagini')

  if fnames == []:
    raise Exception('Niente immagini! Il path è sbagliato, magari ext è sbagliato! ')
    logger.exception('Non ho trovato immagini in {dataset_path_mass}')

  else:
    pass
  masknames = glob.glob(os.path.join(dataset_path_mask, f"*.{ext}"))
  logger.info(f'ho analizzato {dataset_path_mask} cercando le maschere')

  if masknames==[]:
    raise Exception('Immagini o path non trovati!')
    logger.exception('Non ho trovato maschere in {dataset_path_mask}')

  else:
    pass
  X = []
  Y = []
  class_labels=[]
  for fname in fnames:
    try:
      assert(fname.replace(dataset_path_mass, dataset_path_mask) in masknames)
      logger.debug(f'sto verificando che {fname} sia in {dataset_path_mask}')
      Y.append(fname.replace(dataset_path_mass, dataset_path_mask))
      X.append(fname)

      if benign_label in fname:
        class_labels.append(0)
      elif malign_label in fname:
        class_labels.append(1)
    except:
      warnings.warn(f"Attenzione, per {fname} non vi è corrispondenza in {dataset_path_mask}, controlla che l'immagine non sia corrotta o non sia mancante")
      logger.warning(f"Attenzione, per {fname} non vi è corrispondenza in {dataset_path_mask}, controlla che l'immagine non sia corrotta o non sia mancante")
      pass

  return np.array(X), np.array(Y) , np.array(class_labels)

import pickle
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

def radiomic_dooer(list_test,datapath,endpath,lab,extrc):

  """
  Funzione per estrarre le feature con pyradiomics e salvarle in un dizionario.

  type list_test: lista
  param list_test: lista con path immagine e relativa maschera normalizzata

  type datapath: stringa
  param datapath: percorso cartella dove si trova l'immagine

  type endpath: stringa
  param endpath: cartella dove si salva il pickle del dizionario

  type resize: bool
  param resize: Se TRUE fa il reshape delle maschere per combaciare con quello delle immagini



  """
  b=time.perf_counter()
  try:
    logger.debug(f'sto cercando di estrarre le feature da {list[0]} utilizzando come maschera {list[1]}')
    info=extrc.execute(list_test[0],list_test[1],lab)

  except:
    raise Exception('Problema con pyradiomics: forse vi è un problema col label o i path. Controlla che pyradiomics sia installato e che da radiomics sia importato featureextractor')
    logger.exception(f'Problema con pyradiomics: forse vi è un problema col label o i path per {list[0]} e {list[1]} . Controlla che pyradiomics sia installato e che da radiomics sia importato featureextractor')
  c=time.perf_counter()
  logging.info(f'time to extract:{c-b}')
  d=time.perf_counter()
  pattern=re.compile('[M][\w-]*[0-9]*[\w]{13}')
  logging.debug(f'in regex il pattern è {pattern}')
  name=re.findall(pattern,list_test[0])
  logging.debug(f'il patter che sto cercando è {name}')

  dict_r={name[0]:info}
  try:
    with open(os.path.join(endpath,f'feats_{name[0]}.pickle'), 'wb') as handle:
      pickle.dump(dict_r, handle, protocol=pickle.HIGHEST_PROTOCOL)
      logging.info(f'salvato il pickle feats_{name[0]}.pickle in {endpath}')
  except:
    raise Exception('Qualcosa è andato male nel definire il path di arrivo, controlla che endpath sia giusto')
    logger.exception(f'Qualcosa è andato male nel definire il path di arrivo {datapath}, controlla che {endpath} sia un endpath giusto')
  del(dict_r)
  logger.info(f'time to update:{d-c}')

  return 'time to update:{d-c}'

def read_pgm_as_sitk(image_path):
  """ Read a pgm image as sitk image """
  np_array = np.asarray(PIL.Image.open(image_path))
  logger.info(f'sto leggendo {image_path}')
  sitk_image = sitk.GetImageFromArray(np_array)
  logger.info(f'sto convertendo {image_path}')

  return sitk_image

def dict_update_radiomics(data_path,dictionary):

  """
  Funzione per unire i vari dizionari creati con radiomic_dooer per poi creare il dataframe

  type data_path: stringa (.pickle)
  param data_path: percorso del pickle da aprire

  type dictionary: dizionario
  param dictionary: dizionario generale del dataframe

  """
  with open(data_path, 'rb') as handle:
    logger.info(f'ho aperto {data_path}')
    b = pickle.load(handle)
    logger.info(f'ho caricato {handle}')
    dictionary.update(b)
    logger.info(f'ho fatto un update a {dictionary}')


  return(dictionary)

def blender(img1,img2,a,b):
  """
  Funzione per sovraimporre due immagini con sfumatura

  type img1: array numpy
  param img1: immagine da sovrapporre
  type img2: array numpy
  param img2: immagine da svrapporre
  type a: int or float
  param a: valore di sfumatura di img1
  type b: int or float
  param b: valore di sfumatura di img2
  """
  try:
    image=cv2.addWeighted(img1,a, img2, b,0)
    logger.debug('sto cercando di sovrapporre le immagini con pesi rispettivamente {a} e {b}')
  except:
    raise Exception('Sovrapposizione non riuscita. Controllare che le immagini siano giuste e che a e b siano numeri.')
    logger.exception('Sovrapposizione non riuscita. Controllare che le immagini siano giuste e che a e b siano numeri.')
  logger.info('sovrapposizione riuscita')
  return  image

def dice(pred, true, k = 1):
  """
    Funzione per calcolare l'indice di Dice

    type pred: array numpy
    param pred: immagini predette dal modello

    type true : array numpy
    param true: immagini target

    type k : int
    param k: valore pixel true della maschera
  """

  intersection = np.sum(pred[true==k]) * 2.0
  try:
    dice = intersection / (np.sum(pred) + np.sum(true))
  except ZeroDivisionError:
    logger.exception('provato a dividere per zero!')
  logger(f'calcolato correttamente il dice ottenendo {dice}')
  return dice

def dice_vectorized(pred, true, k = 1):
  """
    Versione vettorizzata per calcolare il coefficiente di dice
    type pred: array numpy
    param pred: immagini predette dal modello

    type true : array numpy
    param true: immagini target

    type k : int
    param k: valore pixel true della maschera
  """
  intersection = 2.0 *np.sum(pred * (true==k), axis=(1,2,3))
  try:
    dice = intersection / (pred.sum(axis=(1,2,3)) + true.sum(axis=(1,2,3)))
  except ZeroDivisionError:
    logger.exception('provato a dividere per zero!')
  logger(f'calcolato correttamente il dice medio ottenendo {dice}')
  return dice

import matplotlib.pyplot as plt
def modelviewer(model):
  """
  Funzione per visualizzare l'andamento della loss di training e validazione per l'autoencoder e per il classificatore
    type model:  model.fit()
    param model: history del modello di Keras ottenuto dalla funzione
  """

  plt.figure('modelviewer')
  plt.subplot(2,1,1)
  plt.title('autoencoder')
  try:
    plt.plot(model.history['decoder_output_loss'])
    plt.plot(model.history['val_decoder_output_loss'])
  except:
    raise Exception('Attenzione, o model non è un modello Keras o il modello non ha i campi decoder_output_loss o val_decoder_output_loss')
    logger.exception('Attenzione, o model non è un modello Keras o il modello non ha i campi decoder_output_loss o val_decoder_output_loss')
  plt.legend(['loss', 'val_loss'])
  plt.subplot(2,1,2)
  plt.title('classifier')
  try:
    plt.plot(model.history['classification_output_loss'])
    plt.plot(model.history['val_classification_output_loss'])
  except:
    raise Exception('Attenzione, il modello non ha i campi classification_output_loss o val_classification_output_loss')
    logger.exception('Attenzione, il modello non ha i campi classification_output_loss o val_classification_output_loss')
  plt.legend(['loss', 'val_loss'])
  plt.show()
  return

import tensorflow as tf

def heatmap(x,model):
  """
  Funzione che mostra la heatmap dell'ultimo layer convoluzionale prima del classificatore senza funzionalità radiomiche
    type x: array numpy
    param x: immagine da segmentare

    type model : keras model
    param model: modello allenato
  """
  img_tensor =x[np.newaxis,...]
  preds = model.predict(img_tensor)[1]
  argmax = np.argmax(preds)
  conv_layer = model.get_layer("last_conv")
  heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

  with tf.GradientTape() as gtape:
      conv_output, predictions = heatmap_model(img_tensor)
      loss = predictions[1][:, np.argmax(predictions[1])]
      grads = gtape.gradient(loss, conv_output)
      pooled_grads = K.mean(grads, axis=(0, 1, 2))

  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
  heatmap = np.maximum(heatmap, 0)
  max_heat = np.max(heatmap)
  if max_heat == 0:
      max_heat = 1e-10
  heatmap /= max_heat


  plt.matshow(heatmap.squeeze())
  plt.show()

  x = np.asarray(255*x, np.uint8)
  heatmap = np.asarray(255*heatmap.squeeze(), np.uint8)


  heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]))



  plt.imshow(blender(x,heatmap,1,1))
  plt.axis('off')
  if argmax==1:
    plt.title('the mass is malign')
  else:
    plt.title('the mass is benign')

  return heatmap

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras import models

def heatmap_rad(x,feature,model):
  """
  Funzione che mostra la heatmap dell'ultimo layer convoluzionale prima del classificatore con funzionalità radiomiche
    type x: array numpy
    param x: immagine da segmentare

    type feature: array numpy
    param feature:feature estratte con pyradiomics

    type model : keras model
    param model: modello allenato
  """
  img_tensor =x[np.newaxis,...]
  feature_tensor=feature[np.newaxis,...]
  preds = model.predict([img_tensor,feature_tensor])[1]
  argmax = np.argmax(preds)
  conv_layer = model.get_layer("last_conv")
  heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

  # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
  with tf.GradientTape() as gtape:
      conv_output, predictions = heatmap_model([img_tensor,feature_tensor])
      loss = predictions[1][:, np.argmax(predictions[1])]
      grads = gtape.gradient(loss, conv_output)
      pooled_grads = K.mean(grads, axis=(0, 1, 2))

  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
  heatmap = np.maximum(heatmap, 0)
  max_heat = np.max(heatmap)
  if max_heat == 0:
      max_heat = 1e-10
  heatmap /= max_heat

  plt.figure('heatmap')
  plt.matshow(heatmap.squeeze())
  plt.show()

  x = np.asarray(255*x, np.uint8)
  heatmap = np.asarray(255*heatmap.squeeze(), np.uint8)


  heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]))

  plt.figure('Heatactivation')
  plt.imshow(blender(x,heatmap,1,1))
  plt.axis('off')
  if argmax==1:
    plt.title('the mass is malign')
  else:
    plt.title('the mass is benign')
  plt.show()
  return heatmap

def plot_roc_curve(fper, tper,auc):
  """
  Funzione che fa il plot della curva roc
  type fper: float
  param fper: percentuale falsi positivi

  type tper: float
  param tper:percentuale veri positivi

  """
  plt.figure('AUC')
  plt.plot(fper, tper, color='orange', label='ROC')
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve with AUC = %.2f'%auc)
  plt.legend()
  plt.show()

from skimage.filters import threshold_multiotsu

def otsu(image,n_items=2):
  """
  Funzione che implementa l'algoritmo di Otsu per la segmentazione
    type image: numpy array
    param fper: immagine da segmentare

    type n_items: intero
    param n_items:numero di oggetti da segmentare nell'immagine

  """
  thresholds = threshold_multiotsu(image,classes=n_items)
  regions = np.digitize(image, bins=thresholds)
  return regions