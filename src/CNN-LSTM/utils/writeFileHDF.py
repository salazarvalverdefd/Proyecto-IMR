import numpy as np
import cv2
import json
import h5py
import os


class WriteFile():

    def __init__(self,
                 data_set,
                 in_dir = "/content/drive/MyDrive/Proyectos-independientes/Proyecto-MINSA/src/",
                 dir_file_data_mp4 = "Data/BRATS-2015/Procesado/MP4/",
                 dir_file_data_np = "Data/BRATS-2015/Procesado/HDF/",
                 dir_json = "CNN-LSTM/utils/dataAugmentation.json"
                 ):

        self.in_dir = in_dir
        self.data_set = data_set
        self.dir_file_data_mp4 = in_dir + dir_file_data_mp4
        self.dir_file_data_np = in_dir + dir_file_data_np
        self.dir_json = in_dir + dir_json

    def __load_set__(self):

        X_train_check = []
        y_train_check = []

        with open(self.dir_json) as archivo:
            data = json.load(archivo)

        if self.data_set == 'train':
            path = 'train'
            X = 'X_train'
            y = 'y_train'
        elif self.data_set == 'test':
            path = 'test'
            X = 'X_test'
            y = 'y_test'
        elif self.data_set == 'validation':
            path = 'valid'
            X = 'X_valid'
            y = 'y_valid'

        for video,label in zip(data[path][X],data[path][y]):
            X_sample = []
            y_sample = []
            path = self.dir_file_data_mp4 + self.data_set + "/" + video
            #print("Path:",path)
            captura = cv2.VideoCapture(path)
            cont = 0
            while (captura.isOpened()):
                ret, img = captura.read()
                img = cv2.resize(img, (90,90), interpolation = cv2.INTER_CUBIC)
                #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #print("Shape:",img.shape)
                X_sample.append(img)
                y_sample.append(label)
                cont = cont + 1
                if cont == 155:
                    print("Se termino de procesar el video:",video)
                    X_train_check = np.array(X_sample)
                    y_train_check = np.array(y_sample)
                    with h5py.File(self.dir_file_data_np + self.data_set +"/" +  video[:-4] + ".h5", "w") as hdf:
                        hdf.create_dataset('data', data = X_train_check)
                        hdf.create_dataset('labels', data = y_train_check)
                    print("Escritura exitosa del video en formato h5!")
                    #X_train_check.append(np.array(X_sample))
                    #y_train_check.append(np.array(label))
                    #print("X_sample shape:",np.array(X_sample).shape)
                    #print("label shape:",np.array(label).shape)
                    break
            captura.release()

        # X_train_check = np.array(X_train_check)
        # y_train_check = np.array(y_train_check)

        # print("shape X_train_check:",X_train_check.shape)
        # print("shape y_train_check:",y_train_check.shape)

        """
            _name = "data" + "_" + str(self.data_set)
            with h5py.File(self.dir_file_data_np + self.data_set +"/" +  _name + ".h5", "w") as hdf:
                hdf.create_dataset('data', data = X_train_check)
                hdf.create_dataset('labels', data = y_train_check)
        """
        return print("Escritura exitosa de todos los videos a formato .H5 !:")

    def __getDataGeneration__(self):
        name = "data" + "_" + str(self.data_set)
        with h5py.File(self.dir_file_data_np + self.data_set + "/" + name + ".h5", "r") as f:
            X_batch = f['data'][:]
            y_batch = f['labels'][:]
        print("Se obtuvo el archivo de data correctamente!")
        return X_batch,y_batch