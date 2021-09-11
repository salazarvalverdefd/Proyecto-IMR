import os
import cv2
import numpy as np
from random import shuffle
import h5py
import sys



def get_frames(current_dir, file_name,_images_per_file = 155 ,img_size = 224 ):
    """
    Funcion que extrae los frames de un video, los cuales retornan
    como un np.array
    
    
    Parameters
    ----------
    current_dir : carpeta donde se encuentran los videos
    file_name :   nombre del archivo de video
    _images_per_file : numero de frames a extraer por video
    img_size : tamaño de la imagen
    
    Returns
    -------
    resul : ndarray que contiene el conjunto de frames por video

    """
    
    in_file = os.path.join(current_dir, file_name)
    
    images = []
    
    vidcap = cv2.VideoCapture(in_file)
    
    success,image = vidcap.read()
        
    count = 0

    while count<_images_per_file: # 155 frames
                
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        res = cv2.resize(RGB_img, 
                         dsize = (img_size, img_size),
                         interpolation = cv2.INTER_CUBIC
                         )
    
        images.append(res) # en este arreglo metemos el total de imagenes que lee la CNN por video
    
        success,image = vidcap.read()
    
        count += 1
        
    resul = np.array(images)
    
    resul = (resul / 255.).astype(np.float16)
        
    return resul 


def label_video_names(in_dir):
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
    
    for _ , _ , file_names in os.walk(in_dir):
      for file_name in file_names:
          
        if file_name[0:3] == 'HGG':
            labels.append([1,0])
            names.append(file_name)
            
        elif file_name[0:3] == 'LGG':
            labels.append([0,1])
            names.append(file_name)      
    
    c = list(zip(names,labels)) 
    shuffle(c)
    names, labels = zip(*c)
            
    return names, labels



def get_transfer_values(current_dir, file_name,transfer_values_size,image_model_transfer,
                        _images_per_file = 155,img_size_touple = (224,224)):
    
    """
    Parameters
    ----------
    current_dir : carpeta donde se encuentran los videos.
    file_name : nombre del video
    _images_per_file: numero de frames extraidos por video
    img_size_touple: tamaño del frame en forma de tupla
    transfer_values_size: tamaño del vector de transferencia (4096)
    image_model_transfer : red CNN definida
    
        
    Returns
    -------
    transfer_values:  vector de tamaño = (155,4096) que corresponde a la extraccion 
    de características de la CNN.
    """
    
    # Forma y tamaño del batch de imágenes (155,224,224,3)
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    # Inicializamos contenedor de batch de imágenes
    image_batch = np.zeros(shape = shape, dtype=np.float16) 
    
    # Obtenemos todo el batch de imégenes desde el video
    image_batch = get_frames(current_dir, file_name)
      
    # Forma y tamaño del vector de salida  de la CNN(155,4096)
    shape = (_images_per_file, transfer_values_size) 
    
    # Inicializamos contenedor de la matriz de salida
    transfer_values = np.zeros(shape=shape, dtype=np.float16)
    
    # Feature Extraction CNN
    transfer_values = image_model_transfer.predict(image_batch)
            
    return transfer_values 



def proces_transfer(vid_names, in_dir, labels, transfer_values_size, 
                    image_model_transfer,_images_per_file = 155 , img_size_touple = (224,224)):
    
    """
    Parameters
    ----------
    vid_names : lista que contiene los nombres de los videos.
    in_dir : directorio en los cuales se encuentra el video
    labels: lista que contiene las categorias de los videos
    _images_per_file: numero de frames extraidos por video
    img_size_touple: tamaño de la imagen 
    transfer_values_size : tamaño del vector de salida de la CNN
    image_model_transfer: instancia de la red CNN
    
    
    Returns
    -------
    transfer_values:vector de características
    labelss: formato de salida
    
    """
    count = 0
    
    # Número de videos en el dataset
    tam = len(vid_names)
    
    # Asignamos previamente input-batch-array a las imagenes (155,224,224,3) 
    shape = (_images_per_file,) + img_size_touple + (3,) 
    
    while count < tam:
        
        # Obteniendo el nombre del video
        video_name = vid_names[count]
        
        # Inicializando el contenedor del batch de imágenes(155,224,224,3)
        image_batch = np.zeros(shape = shape, dtype=np.float16) 
        
        # Obtenemos lso frames del video
        image_batch = get_frames(in_dir, video_name)
        
        # Forma y tamaño del vector de salida  de la CNN(155,4096)
        shape = (_images_per_file, transfer_values_size)
        
        # Inicializamos contenedor de la matriz de salida
        transfer_values = np.zeros(shape = shape, dtype=np.float16) 
        
        
        # Feature Extraction CNN
        transfer_values = image_model_transfer.predict(image_batch) 
         
        # Categoria del video
        labels1 = labels[count]
        
        # Inicializando contenedor
        aux = np.ones([_images_per_file,2])
        
        # Valor final
        labelss = labels1*aux
        
        yield transfer_values, labelss
        
        count+=1 
        
def print_progress(count, max_count):
    
    # Porcentaje de completado.
    pct_complete = count / max_count
    msg = "\r- Progreso: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()



def make_files(n_files,names_training,in_dir,labels_training, transfer_values_size,image_model_transfer):
    """
    

    Parameters
    ----------
    n_files : TYPE
        DESCRIPTION.
    names_training : TYPE
        DESCRIPTION.
    in_dir : TYPE
        DESCRIPTION.
    labels_training : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    gen = proces_transfer(names_training, in_dir, labels_training, transfer_values_size,image_model_transfer)
    

    numer = 1

    # Obtenemos tipos de columnas
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('./prueba.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape = chunk[0].shape, maxshape = maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape = chunk[1].shape, maxshape = maxshape2,
                                 chunks=chunk[1].shape, dtype = chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            numer += 1  


def make_files_test(n_files,names_test,in_dir,labels_test,transfer_values_size,image_model_transfer):
    
    """
    Parameters
    ----------
    n_files : TYPE
        DESCRIPTION.
    names_training : TYPE
        DESCRIPTION.
    in_dir : TYPE
        DESCRIPTION.
    labels_training : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    
    """
    gen = proces_transfer(names_test, in_dir, labels_test,transfer_values_size,image_model_transfer)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('pruebavalidation.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            numer += 1


def process_alldata_training():
    """
    

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    """
    
    joint_transfer=[]
    frames_num=155
    count = 0
    
    with h5py.File('./prueba.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target 


def process_alldata_test():
    """
    

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    """
    
    joint_transfer=[]
    frames_num=155
    count = 0
    
    with h5py.File('./pruebavalidation.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target 






