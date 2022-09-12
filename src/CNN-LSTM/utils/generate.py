import numpy as np
import keras
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence,to_categorical

# https://es.acervolima.com/keras-fit-y-keras-fit_generator

class VideoDataGenerator(Sequence):

    def __init__(self,
                 sample_list,
                 base_dir,
                 batch_size = 1,
                 num_frames = 155,
                 shuffle = True,
                 dim = (90,90),
                 num_channels = 3,
                 num_classes = 2,
                 verbose = 1):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list
        self.on_epoch_end()

    def on_epoch_end(self):
        #Actualizamos indices despues de cada epoca
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Número de batchs por epoca
        return int(np.floor(len(self.sample_list) / self.batch_size))

    def __data_generation(self, list_IDs_temp):

        # Inicializamos el contenedor
        X = np.zeros((self.batch_size, self.num_frames, *self.dim, self.num_channels),
                     dtype = np.float64)
        y = np.zeros((self.batch_size, self.num_frames),
                     dtype = np.float64)

        # Generamos los datos
        for i, ID in enumerate(list_IDs_temp):
            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
            with h5py.File(self.base_dir + ID, 'r') as f:
                X[i] = np.array(f.get("data"))
                # Removemos el tumor comleto(WT)
                y[i] = np.array(f.get("labels"))
        return X, y

    def __getitem__(self, index):

        'Funcion que genera un lote de datos'
        #print("Inicio __getitem__")
        # Generamos indices
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        #print("indexes:",indexes)
        #print("len indexes:",len(indexes))
        # Lista de IDs
        sample_list_temp = [self.sample_list[k] for k in indexes]
        #print("len sample_list_temp",sample_list_temp)
        # Generamos la data final
        X, y = self.__data_generation(sample_list_temp)
        #_y = to_categorical(y, num_classes = self.num_classes)
        #print("_y:",_y)
        #print("_y keras shape:",_y.shape)
        #print("X Shape:",X.shape)
        #print("y Shape:",y.shape)
        #print("Fin __getitem__")
        return X,y
