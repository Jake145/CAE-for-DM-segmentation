import keras
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Classes.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
class MassesSequence(keras.utils.Sequence):
    """ Classe per fare data augmentation per CAE """

    def __init__(self, x, y,label_array, img_gen, batch_size=10, shape=(124,124)):
        """

        Parametri:

        x (np.array): immagini
        y (np.array): maschere
        label_array (np.array): label di classificazione (benigno o maligno)
        batch_size (int): dimensione della batch
        img_gen (ImageDatagenerator): istanza della classe ImageDatagenerator
        shape (tuple): dimensione delle immagini. Di default (124, 124)

        """
        self.x, self.y,self.label_array = x, y,label_array
        self.shape = shape
        self.img_gen = img_gen
        self.batch_size = batch_size


    def __len__(self):
        return len(self.x) // self.batch_size

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        self.x, self.y ,self.label_array= shuffle(self.x, self.y,self.label_array)

    def process(self, img, transform):
        """ Apply a transformation to an image """
        img = self.img_gen.apply_transform(img, transform)
        return img

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_array = self.label_array[idx * self.batch_size:(idx + 1) * self.batch_size]

        X=[];
        Y=[];
        Classes=[];

        for image, mask,label in zip(self.x, self.y,self.label_array):
            transform = self.img_gen.get_random_transform(self.shape)
            X.append(self.process(image, transform))
            Y.append(self.process(mask, transform)>0.2)
            Classes.append(label)



        return np.asarray(X,np.float64), [np.asarray(Y,np.float64) ,np.asarray(Classes,np.float)]

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


class MassesSequence_radiomics_big(keras.utils.Sequence):
    """ Classe per data augmentation per CAE con grandi dati """


    def __init__(self, x, y,label_array,features, img_gen, batch_size=5, shape=(2048, 1536),shape_tensor=(2048, 1536, 1)):
        """ Inizializza la sequenza

        Parametri:

        x (np.array): path delle immagini
        y (np.array): path delle maschere
        label_array (np.array): label di classificazione
        features (np.array): array di feature dopo la pca
        batch_size (int): dimensione della batch
        img_gen (ImageDatagenerator): Una istanza della classe ImageDatagenerator
        shape (tuple): shape dell'immagine. Di Default è (2048, 1536) per il limite di colab, per la Unet invece è metà di queste.

        """
        self.x, self.y,self.label_array,self.features = x, y,label_array,features
        self.shape = shape
        self.shape_tensor=shape_tensor
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

        for image, mask,label,feature in zip(batch_x, batch_y,batch_label_array,batch_features):
            transform = self.img_gen.get_random_transform(self.shape)
            X_el=resize(imread( str(image)), self.shape_tensor)
            Y_el=resize(imread( str(mask)), self.shape_tensor)
            X.append(self.process(X_el, transform))
            del(X_el)
            Y.append(self.process(Y_el, transform))
            del(Y_el)
            Classes.append(label)
            Features.append(feature)

        return [np.array(X)/255,np.asarray(Features,np.float64)], [np.array(Y)/255 ,np.asarray(Classes,np.float)]
