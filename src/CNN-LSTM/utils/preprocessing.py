import os
import cv2
import numpy as np
from random import shuffle, Random
import h5py
import sys
from sklearn.model_selection import train_test_split
from torchvision import transforms
from skimage.util import random_noise
import json
import torch
import uuid
import random

class PrepareData:

    def __init__(self, current_dir, images_per_file, img_size):
        """

        Parameters
        ----------
        current_dir : TYPE
            carpeta donde se encuentran los videos
        images_per_file : TYPE
           numero de frames a extraer por video
        img_size : TYPE
           tamaño de la imagen

        Returns
        -------
        None.

        """

        self.current_dir = current_dir
        self.images_per_file = images_per_file
        self.img_size = img_size

    def get_frames(self, file_name):
        """
        Funcion que extrae los frames de un video, los cuales retornan
        como un np.array
        Parameters
        ----------
        current_dir : carpeta donde se encuentran los videos
        Returns
        -------
        resul : ndarray que contiene el conjunto de frames por video

        """

        in_file = os.path.join(self.current_dir + "train/", file_name)

        images = []

        vidcap = cv2.VideoCapture(in_file)

        success, image = vidcap.read()

        count = 0

        while count < self.images_per_file:
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            res = cv2.resize(
                RGB_img, dsize=(self.img_size, self.img_size),
                interpolation = cv2.INTER_CUBIC
            )

            images.append(
                res
            )  # en este arreglo metemos el total de imagenes que lee la CNN por video

            success, image = vidcap.read()

            count += 1

        resul = np.array(images)

        resul = (resul / 255.0).astype(np.float16)

        return resul


    def label_video_names(self):
        """
        Funcion que categoriza los datos(videos) de acuerdo al nombre que llevan

        Parameters
        ----------
        in_dir : carpeta donde se encuentran los videos.

        Returns
        -------
        names : lista que contiene los nombres de los videos
        labels : lista que contiene la categoria de los video,
        ya sea (1 = HGG) o (0=LGG).

        """
        names = []
        labels = []

        for _, _, file_names in os.walk(self.current_dir):
            for file_name in file_names:

                if file_name[0:3] == "HGG":
                    labels.append([1, 0])
                    names.append(file_name)

                elif file_name[0:3] == "LGG":
                    labels.append([0, 1])
                    names.append(file_name)

        c = list(zip(names, labels))
        Random(1234).shuffle(c)
        names, labels = zip(*c)

        return names, labels


    def split_validation(self,names,labels):

        names = list(names)
        labels = list(labels)

        X_train, X_test, y_train, y_test = train_test_split(names,
                                                            labels,
                                                            test_size = 0.20,
                                                            stratify = labels,
                                                            random_state = 42
                                                            )
        return X_train, X_test, y_train, y_test


class AddGaussianNoise(object):

    def __init__(self, mean = 0., std = None):
        
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.std == None:
            stds = list((np.arange(0, 0.25,0.01, dtype = np.float64)))
            self.std = random.choice(stds)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MyRandomGammaCorrection(object):

    def __init__(self, gamma = None):
        self.gamma = gamma

    def __call__(self,image):

        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0,0,0,0.5,1,1.5]
            self.gamma = random.choice(gammas)
        #print(self.gamma)

        if self.gamma == 0:
            return image
        else:
            return transforms.functional.adjust_gamma(image, self.gamma, gain=1)


class VideoAugmentation():

  def __init__(self,dir,set_data,dir_json):
    self.dir = dir
    self.set_data = set_data
    self.dir_json = dir_json


  def Normalizar(self,f):
    faux = np.ravel(f).astype(float)
    minimum = faux.min()
    maximum = faux.max()
    g = (faux-minimum)*(255) / (maximum-minimum)
    r = g.reshape(f.shape).astype(np.uint8)
    return(r)

  def Mytransforms(self,image):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomAffine((30,70)),
                                    MyRandomGammaCorrection(),
                                    transforms.RandomApply([AddGaussianNoise(0)], p = 0.5)
                                  ])
    tensor_img = transform(image)
    img = tensor_img.permute(1, 2, 0)
    img = img.numpy()
    img = cv2.normalize(img,
                        dst = None,
                        alpha = 0,
                        beta = 255,
                        norm_type = cv2.NORM_MINMAX,
                        dtype = cv2.CV_8U)
    
    #img = img.astype(np.uint8)
    #img = self.Normalizar(img)
    return img

  def unico(x,L):
      esUnico = True
      for i in range(len(L)):
        if x == L[i]:
            esUnico = False
            break
      return esUnico

  def get_file(self):
      with open(self.dir_json) as archivo:
          data = json.load(archivo)
      return data

  def generate_name(self,names_reg):
      _data = self.get_file()[self.set_data]
      X = _data[self.set_data][str(X) + str("_") + str(self.set_data)]
      if self.set_data == 'validation':
          X = _data['valid'][str(X) + str("_") + str("valid")]
      names_reg = X
      return names_reg

  def read_and_write_video(self):

    VIDEO_PATH = self.dir + self.set_data
    names_reg_vid = []
    # names_reg_vid = self.generate_name(names_reg_vid)
    # Leyendo las direcciones de los videos

    for video in os.listdir(VIDEO_PATH):
      split_name = video.split(".")
      link = VIDEO_PATH + "/" + video
      captura = cv2.VideoCapture(link)

      # Extrayendo los frames de video
      # img_array = []
      width = 240
      height = 240
      name = '.'.join(split_name[0:5]) + str(".") + str(uuid.uuid1().hex) + str(".mp4")
      video = cv2.VideoWriter(VIDEO_PATH + "/" + name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height),0)
      cont = 0
      while (captura.isOpened()):
        ret, imagen = captura.read()
        #print("Tipo de imagen:",type(imagen))
        #print("Imagen:",imagen)
        img = self.Mytransforms(imagen)
        #img_array.append(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        video.write(img)
        cont = cont + 1
        if cont == 155:
            break
      video.release()
      captura.release()
      print("Se termino de procesar el video:",name)
    print("Se terminaron de procesar todos los videos")
