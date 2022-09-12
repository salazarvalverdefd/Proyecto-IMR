

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import json
import tensorflow as tf
from keras.layers import Input
from keras import Sequential
from keras.layers import Dense, LSTM,Flatten, TimeDistributed, Conv2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


class DefineModelTimeDistributed():


    def __init__(self,in_dir = "/content/drive/MyDrive/Proyectos-independientes/Proyecto-MINSA/src/CNN-LSTM/",
                 include_top = False, weights = None, input_tensor = None, input_shape = (90,90,3),
                 number_frames = 155, loss = "binary_crossentropy", optimizer = "adam", batch_size = 2,
                 epochs = 10 ,steps_per_epoch = 3,validation_steps = 3, metrics = ['accuracy'], path_weights = "weights/", path_logs = "logs/"):
        
        self.in_dir = in_dir
        self.include_top = include_top
        self.weights = weights
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.number_frames = number_frames
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.metrics = metrics
        self.path_weights = in_dir  + path_weights
        self.path_logs = in_dir + path_logs

    def __call__(self):

        vggnet = tf.keras.applications.VGG16(
                            include_top = self.include_top,
                            weights = self.weights,
                            input_tensor = self.input_tensor,
                            input_shape = self.input_shape
                            )
        print(vggnet.summary())

        return vggnet

    def __matchModels__(self,model_cnn):

        input_shape_time = list(self.input_shape)
        input_shape_time.insert(0,self.number_frames)

        model = Sequential()
        model._name = "VGG16_with_LSTM"
        model.add(TimeDistributed(model_cnn, input_shape = input_shape_time))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100, activation = 'relu', return_sequences = False))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dropout(.5))
        model.add(Dense(1, activation = 'sigmoid'))

        print(model.summary())

        return model

    def __compile__(self,model):

        model.compile(loss = self.loss,
                      optimizer = self.optimizer,
                      metrics = self.metrics
                    )
        return model

    def __callback__(self):

        model_checkpoint = ModelCheckpoint(filepath = self.path_weights + "model-{epoch:04d}-{val_loss:.4f}.hdf5",
                                            monitor = 'val_loss',
                                            verbose = 1,
                                            save_best_only = False,
                                            #save_weights_only=False,
                                            save_freq = 'epoch'
                                          )
        csv_logger = CSVLogger(self.path_logs + "model-{epoch:04d}-{val_loss:.4f}", append = True)

        return [model_checkpoint,csv_logger]


    def __training__(self,model,train_generator,valid_generator):

        print("Fit model on training data")

        history = model.fit(train_generator,
                            batch_size = self.batch_size,
                            steps_per_epoch = self.steps_per_epoch,
                            epochs = self.epochs,
                            callbacks = [self.__callback__()[0],self.__callback__()[1]],
                            use_multiprocessing = True,
                            validation_data = valid_generator,
                            validation_steps = self.validation_steps
                           )
        return [history,model]

    def __PlotResults__(self,history):

        fig = plt.gcf()
        fig.set_size_inches(12, 8)

        # plot loss
        plt.subplot(2,2,1)
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.plot(history.history['loss'], color = 'blue', label = 'train')
        plt.plot(history.history['val_loss'],color = 'orange',label = 'validation')
        plt.legend(['entrenamiento', 'validación'], loc = 'upper left')

        # plot accuracy
        plt.subplot(2,2,2)
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
        plt.plot(history.history['val_accuracy'], color = 'orange', label = 'validation')
        plt.legend(['Train', 'Validation'], loc = 'upper left')
        plt.show()

        return print("Mostrando graficas...")


    def __evaluateTest__(self,model,x_test,y_test,batch_size):
        print("Evaluate on test data")
        results = model.evaluate(x_test, y_test, batch_size = batch_size)
        print("test loss, test acc:", results)

    def __prediction__(self,model,x_test):
        print("Generate predictions for 3 samples")
        predictions = model.predict(x_test[:3])
        print("predictions:", predictions)



