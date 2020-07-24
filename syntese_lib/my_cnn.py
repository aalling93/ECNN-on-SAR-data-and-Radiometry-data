import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import os.path
import tensorflow as tf
import sys
import pydot
import graphviz 
import warnings
import glob
import tensorflow_hub as hub
import math
import time

from tensorflow import keras
from keras import layers
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model, Input
from keras.engine import training
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List

import my_s1 as s1
import my_tools as tools
import my_s3 as s3

input_shape = (200,200,1)
def s1_model(model_input: input_shape,Name,dropout=0.5,regularisation=0.005,filter_size=3,stride=2) -> training.Model:
  '''
    This function makes a simple CNN model for a grayscale img.
    The function utilizes the theory described in the theory section in the report corresponding to the report, to make a single CNN model.
    The model has the convolutional layers, the bactch normalization, the ReLu activation, the MaxPooling method, and a dropout layer. Also, the FC layer (called Dense layer in Keras). 
    The Function has not been optimized, i.e. the paramters havent been iterated several times.

    This funciton only defines the model and the type of training data. It is NOT training the model. For the function on how to train the model, see cnn.training_CNN()

    The function can be used to train on all types of grayscale images of correct size (200,200,1).

    Inputs:
      input_shape: The shape of the input when defining the model.
      Name[str]:string of the name of the model
      dropout[float]: float value between 0 and 1. Probability for dropout, see the report.
      regularisation[float]:float value of the input layer, bias and outpyt layer bias, see the report.
      filter_size[int]: interger value of filter size. Must be odd number. If e.g 3 is put, (3x3) filters will be used, see the report.
      stride[int]: stride for filter, see the report.

    Outputs:
      keras.Model: the defined model with the name.
    
    Example:
      s1_model((200,200) + (1,),'cnn_mode1',dropout=0.5,regularisation=0.005,filter_size=12,stride=12)

    Author:
      Kristian Soerensen
      July 2020
      kristian.sorensen@hotmail.com
  '''
  tf.autograph.set_verbosity(0)
  #defining input shape. s1 is (200,200,1) and s3 is (200,200,3)..
  inputs = keras.Input(shape=input_shape)
  # Image augmentation block can be used...
  #x = dataset_augmentation(inputs)
  # 1st block ---------------------------------
  #adding a convolutional layer, https://keras.io/layers/convolutional/
  x = keras.layers.Conv2D(32, filter_size, strides=stride, padding='same')(inputs)
  x = keras.layers.BatchNormalization()(x)
  #adding activation function
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling2D(filter_size, strides=stride, padding='same')(x)
  # 2nd block
  x = keras.layers.Conv2D(64, filter_size, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling2D(filter_size, strides=stride, padding='same')(x)
  # Now, making several such "Blocks" where the sizes of the filters are changed...  I.e. adding 4 blocks more here.
  for size in [82, 96]:
    #this "block" of layers (convolutional layer, activation, pooling and regularization) can now be applied a number of times.... i.e copy pastye..
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.SeparableConv2D(size, 3, padding='same')(x)
    x = keras.layers.Conv2D(size, filter_size, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    #adding Maxpooling as described in the theory and https://keras.io/layers/pooling/. Default stride is the same as pool..
    x = keras.layers.MaxPooling2D(filter_size, strides=stride, padding='same')(x)


  # last block ---------------------------------
  x = keras.layers.Conv2D(108, filter_size, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.GlobalAveragePooling2D()(x)
  # activation fucntion used for two class classification... see report...
  activation = 'sigmoid'
  units = 1
  #regularization, https://keras.io/layers/core/
  x = keras.layers.Dropout(dropout)(x)
  #Flattens the input. Does not affect the batch size needed for FC... see https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
  #Just your regular densely-connected NN layer, i.e. the FC...adding different weight regularizers. Lastly,activation function to classify...
  outputs = keras.layers.Dense(units, kernel_regularizer=regularizers.l2(regularisation),bias_regularizer=regularizers.l2(regularisation),activity_regularizer=regularizers.l2(regularisation),activation=activation)(x)

  return keras.Model(inputs, outputs, name=Name)

input_shape_s3 = (200,200,3)
def s3_model(model_input: input_shape_s3,Name,dropout=0.5,regularisation=0.005,filter_size=3,stride=2) -> training.Model:
  '''
    This function makes a CNN model for a 3-channel img.
    The function utilizes the theory described in the theory section in the report corresponding to the report, to make a single CNN model.
    The model has the convolutional layers, the bactch normalization, the ReLu activation, the MaxPooling method, and a dropout layer. Also, the FC layer (called Dense layer in Keras). 
    
    The model utilizes residuals in otder to make micro arcitectures within the network, moticated from ResNET. These can learn features in the data on different scales and combined the results.
    Moreover, as the spatial volume decreases, the filter size increases. This is comon practive. 

    This funciton only defines the model and the type of training data. It is NOT training the model. For the function on how to train the model, see cnn.training_CNN()

    The function can be used to train on all types of RGB images of correct size (200,200,3).

    Inputs:
      input_shape: The shape of the input when defining the model.
      Name[str]:string of the name of the model
      dropout[float]: float value between 0 and 1. Probability for dropout, see the report.
      regularisation[float]:float value of the input layer, bias and outpyt layer bias, see the report.
      filter_size[int]: interger value of filter size. Must be odd number. If e.g 3 is put, (3x3) filters will be used, see the report.
      stride[int]: stride for filter, see the report.

    Outputs:
      keras.Model: the defined model with the name.
    
    Example:
      s3_model((200,200) + (3,),'cnn_mode3',dropout=0.5,regularisation=0.005,filter_size=9,stride=9)
    
    Example:
      model_s3_13 = s3_model((200,200) + (3,))

    Author:
      Kristian Sørensen
      July 2020
      kristian.sorensen@hotmail.com
  '''
  tf.autograph.set_verbosity(0)
  #defining input shape. s1 is (200,200,1) and s3 is (200,200,3)..
  inputs = keras.Input(shape=input_shape_s3)
  # 1st block ---------------------------------
  #adding a convolutional layer, https://keras.io/layers/convolutional/
  x = keras.layers.Conv2D(16, filter_size, strides=stride, padding='same')(inputs)
  x = keras.layers.BatchNormalization()(x)
  #adding activation function
  x = keras.layers.Activation('relu')(x)
  # 2nd block
  x = keras.layers.Conv2D(32, filter_size, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  # residuals..
  residuals = x  
  # Now, making several such "Blocks" where the sizes of the filters are changed...  I.e. adding 4 blocks more here.
  for size in [32, 64, 84, 92]:
    #this "block" of layers (convolutional layer, activation, pooling and regularization) can now be applied a number of times.... i.e copy pastye..
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.SeparableConv2D(size, filter_size, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation('relu')(x)
    x = keras.layers.SeparableConv2D(size, filter_size, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    #adding Maxpooling as described in the theory and https://keras.io/layers/pooling/. Default stride is the same as pool..
    x = keras.layers.MaxPooling2D(filter_size, strides=stride, padding='same')(x)
    
    # Project residual
    residual = keras.layers.Conv2D(size, 1, strides=stride, padding='same')(residuals)
    #add residuls..
    x = keras.layers.add([x, residual])
    # residuals..
    residuals = x 

  # last block ---------------------------------
  x = keras.layers.SeparableConv2D(124, filter_size, padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.GlobalAveragePooling2D()(x)
  # activation fucntion used for two class classification... see report...
  activation = 'sigmoid'
  units = 1
  #regularization, https://keras.io/layers/core/
  x = keras.layers.Dropout(dropout)(x)
  #Flattens the input. Does not affect the batch size needed for FC... see https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
  #Just your regular densely-connected NN layer, i.e. the FC...adding different weight regularizers. Lastly,activation function to classify...
  outputs = keras.layers.Dense(units, kernel_regularizer=regularizers.l2(regularisation),bias_regularizer=regularizers.l2(regularisation),activity_regularizer=regularizers.l2(regularisation),activation=activation)(x)

  return keras.Model(inputs, outputs, name=Name)

def training_CNN(model: training.Model, train, val, num_epochs: int, save=True,learning_rate=0.5,verbose=0,batch_size=32) -> Tuple [History, str]:
  '''
  cnn.training_CNN()
  This function takes the model arcitecture and compiles it using the SDG optimiser and the Binary Cross entropy loss-function.
  
  The function defines the optimzer, the Stocastich Gradient Descent using the learning rate. The Learning rate is described in the thesis. 
  For information on the two hyper parameters momentum and decay see https://keras.io/api/optimizers/.
  
  The funcition then compiles the model with all the hyper parameters. Here, callback are defined. These are used for plotting, saving ect. 
  Then, the function is trained on the training data and validataed on the validation data.
  
  Input:
      model: the saved model arcitecture from keras. Made using e.g. cnn.s3_model or cnn.s1_model.
      train: Training data. Made using e.g. cnn.make_dataset
      val: Validataion data. Made using e.g. cnn.make_dataset. 
      num_epochs: Number of epochs
      save[True/false]: If true, the model will be saved to disk. 
      learning_rate[float]: Value between 0 and 1. 
      verbose[int]: if verbose =0, nothing will be printed. If verbose=1, information will be printed.
      batch_size: hyper parater used in keras.

  Output:
      history: object containing the training history, the weight, plots and more
      weights: the weights as an array.

  Example:
      history, weights = training_CNN(model, train_data, val_data, 40, save=True,0.001,verbose=1)
      history, weights = training_CNN(model, train_data, val_data, 40, save=True,0.9,verbose=1)

  Author:
    Kristian Soerensen
    July 2020
    kristian.sorensen@hotmail.com
  ''' 
  #printing info if verbose >0
  if verbose>0:
    print(model.name+' is being trained.\n')
  #setting tensorflow verbosity.
  tf.autograph.set_verbosity(0)
  #defining learning and loss fucntion. 
  optimizer1 = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.01)
  #compiling the model with hyper paramters.
  model.compile(optimizer=optimizer1,loss="binary_crossentropy",metrics=["accuracy","AUC",tf.keras.metrics.BinaryCrossentropy(),tf.keras.metrics.Precision(),tf.keras.metrics.BinaryAccuracy()])
  #path to save weight to..
  filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
  #checkpoint and tensorbord are used in the fitting to get info..
  checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=verbose, save_weights_only=True,save_best_only=True, mode='auto')
  tensor_board = keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=0)
  #fitting the data to the model!
  history = model.fit(train,batch_size=batch_size,epochs=num_epochs, verbose=verbose, callbacks=[checkpoint, tensor_board], validation_data=val)
  weight_files = glob.glob(os.path.join(os.getcwd(), 'weights/*'))
  #making a folder to save models to.
  if os.path.exists(model.name)==False:
    os.mkdir(model.name)
  #saving models and weights..
  if save==True:
    # convert history dict to pandas to save
    hist_df = pd.DataFrame(history.history) 
    # save to json:  
    hist_json_file = model.name+'/history.json' 
    #with open(os.path.join(dirName, '') + 'index.html', 'w') as write_file:
    with open(hist_json_file, mode='w') as f:
      hist_df.to_json(f)
    hist_csv_file = model.name+'/history.csv'
    with open(hist_csv_file, mode='w') as f:
      hist_df.to_csv(f)
    np.save(model.name+'/history.npy',history.history)
    #saving entire model! here, we also have the weights and all!!!
    model.save(model.name+"/model_try.h5")
    print("Model is saved.")

  return history, weight_files

def prediction_image(folder_img='/content/drive/My Drive/syntese/data/cnn/hele_billede',folder_models='/content/drive/My Drive/syntese/data/cnn/models/new',verbose=0):
  '''
    cnn.prediction_image():

    This function predicts if there are fires in an Sentinel-1/Sentinel-3 image pair using a ECNN.

    The function loads the five different CNN models, as explained in the Report.
    It then loads the Prepared Sentinel-1 and Sentinel-3 image. It then split the images up, into the models, i.e.
    The Sentinel-1 image is divided into the VV (for model 1) and the VH (for model2) images. The Sentinel-3 image is divided into S13, S46 and S79.
    
    Each image is then divided into sub images. For the Sentinel-1 images, they are divided into (200,200,1) sub sets. The Sentinel-3 images are divided into (200,200,3) sub sets.
    Each sub set is then put through its respective model (Sentinel-1 VV sub set thourgh model 1 ect.). 
    For each image, a prediction mask is made. If a fire is preset in the sub set, a flag is made in the mask.

    A fire mask is then made for each image.

    The value in the mask corresponds to a prediction for each (200,200) subset. The first 2D array of the mask is 1/0 masks. The second 2D array holds the prediction.

    Input:
      folder_img[str]: Path of folder with image pair. Only the S3 and S1 image are allowed to be in the folder. These must have been prepared according to the report.
      folder_models[str]: Path of folder for models. The folder must have 5 model folder (with names model1, model2 ect), each with a model file called 'model_try.h5'
      verbose[int]: integer for information. Verbose = 0 prints no information. Verbose=1 prints information

    Output:
      masks[list]: A list with 6 prediction masks with:[prediction_model1, prediction_model2, prediction_model3, prediction_model4, prediction_model5, prediction_model_ecnn]. Each maks has a binary and a probability map.
      s1_extend[list]: just the 4 coordinate corners. For e.g. plotting.
      

    Example:
      predict = prediction_image(folder_img='/content/drive/My Drive/syntese/data/cnn/hele_billede',folder_models='/content/drive/My Drive/syntese/data/cnn/models/new',verbose=1)

    Author:
      Kristian Soerensen
      July 2020
      kristian.sorensen@hotmail.com
  '''
  # initiating time is takes.
  start_time = time.time()
  #setting tensorflow verbosity.
  tf.autograph.set_verbosity(0)
  #if verbose>0 printing
  if verbose>0:
    print('\nFire Detection maps are being calculated:\n')

  #loading the models. These should be placed in individual folder as decribed in the thesis.
  model1_path=folder_models+'/model1/model_try.h5'
  model2_path=folder_models+'/model2/model_try.h5'
  model3_path=folder_models+'/model3/model_try.h5'
  model4_path=folder_models+'/model4/model_try.h5'
  model5_path=folder_models+'/model5/model_try.h5'
  models = []
  #loading the models into a list of models.
  models.append(tf.keras.models.load_model(model1_path,custom_objects={'KerasLayer':hub.KerasLayer}))
  models.append(tf.keras.models.load_model(model2_path,custom_objects={'KerasLayer':hub.KerasLayer}))
  models.append(tf.keras.models.load_model(model3_path,custom_objects={'KerasLayer':hub.KerasLayer}))
  models.append(tf.keras.models.load_model(model4_path,custom_objects={'KerasLayer':hub.KerasLayer}))
  models.append(tf.keras.models.load_model(model5_path,custom_objects={'KerasLayer':hub.KerasLayer}))
  #getting the path of image.
  img_paths = tools.file_paths(folder_img,verbose=1,file_type='tif')
  #Only single image pairs are allowed. Use cnn.prediction_folder() for more images.
  assert (len(img_paths)<3),'only image a single pairs is allowed'
  #loading images
  if img_paths[0].split("/")[-1][0:2]=='S3' or img_paths[0].split("/")[-1][0:2]=='s3':
    s3_img,s3_raster,s3_extend, s3_names, s3_extra = s3.get_img([img_paths[0]],data_type='ENVI')
    s1_img,s1_raster,s1_extend, s1_names, s1_extra = s1.get_img([img_paths[1]],data_type='ENVI')
  else:
    s3_img,s3_raster,s3_extend, s3_names, s3_extra = s3.get_img([img_paths[1]],data_type='ENVI')
    s1_img,s1_raster,s1_extend, s1_names, s1_extra = s1.get_img([img_paths[0]],data_type='ENVI')

  
  #ignoring annoying rescaling warnings.
  warnings.filterwarnings("ignore")
  #making the band subsets, and rescaling to [0,1] for the network.
  s1_vv=s1_img[0]
  for j in range(len(s1_vv)):
    s1_vv[j] = ((s1_vv[j]-s1_vv[j].min())/(s1_vv[j].max()-s1_vv[j].min() )*1)
  #s1_vv = np.array([s1_vv])

  s1_vh=s1_img[1]
  for j in range(len(s1_vh)):
    s1_vh[j] = ((s1_vh[j]-s1_vh[j].min())/(s1_vh[j].max()-s1_vh[j].min() )*1)

  #S3, F2, F1
  s3_13=[s3_img[3],s3_img[1],s3_img[0]]
  for j in range(len(s3_13)):
    s3_13[j] = ((s3_13[j]-s3_13[j].min())/(s3_13[j].max()-s3_13[j].min() )*1)
  s3_13 = np.dstack((s3_13[0], s3_13[1],s3_13[2]))

  #S5, F2, F1
  s3_46=[s3_img[4],s3_img[1],s3_img[0]]
  for j in range(len(s3_46)):
    s3_46[j] = ((s3_46[j]-s3_46[j].min())/(s3_46[j].max()-s3_46[j].min() )*1)
  s3_46 = np.dstack((s3_46[0], s3_46[1],s3_46[2]))

  #S5, F2, S8
  s3_79=[s3_img[4],s3_img[1],s3_img[2]]
  for j in range(len(s3_79)):
    s3_79[j] = ((s3_79[j]-s3_79[j].min())/(s3_79[j].max()-s3_79[j].min() )*1)
  s3_79 = np.dstack((s3_79[0], s3_79[1],s3_79[2]))
  warnings.filterwarnings("default")
    
  #size of areas. This is best kept as hardcoded and not changed unles sure!
  size = 200
  half = int(math.ceil(size/2))
  rows, colums = s1_vv.shape

  #making the map for model1
  if verbose>0:
    print('\n Making predictions for model 1')
  
  #array used for the binary maps
  mask_vv_1 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  #array used for the probability maps
  mask_vv_2 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  #stacking maps to 3D array.
  mask_vv =np.dstack((mask_vv_1, mask_vv_2))
  #placeholder for counting
  row = 0
  # first subset is [0:200,0:200]. Second is [100:300,0:200] ect. Each subset is examined for fire.
  for i in range(half,rows-half,half):
    row = row+1
    column =0
    for j in range(half,colums-half,half):
      column = column+1
      #the subset in question
      temp_img =  s1_vv[i-half:i+half,j-half:j+half]
      # adding extra dimension so it is the same dimension as the traing data (label)
      image1 = np.expand_dims(temp_img,axis=0)
      #getting prediction from model 1.
      predictions = (models[0].predict(image1))[0]
      #adding the probability
      mask_vv[row,column,1] = predictions
      if predictions>0.5:
        #chaing binary mask to 1 if probability is above 50%.
        mask_vv[row,column,0] = 1
        
  #doing the same for the other models.
  if verbose>0:
    print('\n Making predictions for model 2')
  #making the mask for model 2
  mask_vh_1 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_vh_2 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_vh = np.dstack((mask_vh_1, mask_vh_2))

  row = 0
  for i in range(half,rows-half,half):
    row = row+1
    column =0
    for j in range(half,colums-half,half):
      column = column+1
      temp_img =  s1_vh[i-half:i+half,j-half:j+half]
      image1 = np.expand_dims(temp_img,axis=0)
      predictions = (models[1].predict(image1))[0]
      mask_vh[row,column,1] = predictions
      if predictions>0.5:
        mask_vh[row,column,0] = 1

  if verbose>0:
    print('\n Making predictions for model 3')
  #making the mask for model 3
  mask_s3_13_1 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_s3_13_2 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_s3_13 = np.dstack((mask_s3_13_1, mask_s3_13_2))

  row = 0
  for i in range(half,rows-half,half):
    row = row+1
    column =0
    for j in range(half,colums-half,half):
      column = column+1
      temp_img =  s3_13[i-half:i+half,j-half:j+half,:]
      image1 = np.expand_dims(temp_img,axis=0)
      predictions = (models[2].predict(image1))[0]
      mask_s3_13[row,column,1] = predictions
      if predictions>0.5:
        mask_s3_13[row,column,0] = 1
        
  if verbose>0:
    print('\n Making predictions for model 4')
  #making the mask for model 4

  mask_s3_46_1 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_s3_46_2 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_s3_46 = np.dstack((mask_s3_46_1, mask_s3_46_2))

  row = 0
  for i in range(half,rows-half,half):
    row = row+1
    column =0
    for j in range(half,colums-half,half):
      column = column+1
      temp_img =  s3_46[i-half:i+half,j-half:j+half,:]
      image1 = np.expand_dims(temp_img,axis=0)
      predictions = (models[3].predict(image1))[0]
      mask_s3_46[row,column,1] = predictions
      if predictions>0.5:
        mask_s3_46[row,column,0] = 1
        
  if verbose>0:
    print('\n Making predictions for model 5')
  #making the mask for model 5

  mask_s3_79_1 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_s3_79_2 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_s3_79 = np.dstack((mask_s3_79_1, mask_s3_79_2))

  row = 0
  for i in range(half,rows-half,half):
    row = row+1
    column =0
    for j in range(half,colums-half,half):
      column = column+1
      temp_img =  s3_79[i-half:i+half,j-half:j+half,:]
      image1 = np.expand_dims(temp_img,axis=0)
      predictions = (models[4].predict(image1))[0]
      mask_s3_79[row,column,1] = predictions
      if predictions>0.5:
        mask_s3_79[row,column,0] = 1
      
  if verbose>0:
    print('\n Making predictions for the ECNN')
  mask_ecnn_1 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_ecnn_2 =np.zeros((math.ceil((rows-half)/half),math.ceil((colums-half)/half)))
  mask_ecnn = np.dstack((mask_ecnn_1, mask_ecnn_2))

  #The ECNN prediction is found as the average of the fine others.
  for i in range(math.ceil((rows-half)/half)):
    for j in range(math.ceil((colums-half)/half)):
      prediciton = (mask_vv[i,j,1]+mask_vh[i,j,1]+mask_s3_13[i,j,1]+mask_s3_46[i,j,1]+mask_s3_79[i,j,1])/5
      mask_ecnn[i,j,1] = prediciton
      if prediciton>0.5:
        mask_ecnn[i,j,0]=1

  #removing the edge. It might make som problems..
  mask_vv = mask_vv[1:,1:,:]
  mask_vh = mask_vh[1:,1:,:]
  mask_s3_13 = mask_s3_13[1:,1:,:]
  mask_s3_46 = mask_s3_46[1:,1:,:]
  mask_s3_79= mask_s3_79[1:,1:,:]
  mask_ecnn = mask_ecnn[1:,1:,:]

  #if there is a 0 in the masks, a fire has been located.  
  if 0 in mask_ecnn[:,:,0]:
    print('(get_predictions): A fire has been located')
  
  #returning all maps.
  maps = [mask_vv,mask_vh,mask_s3_13,mask_s3_46,mask_s3_79,mask_ecnn]
  if verbose>0:
    print("--- %s seconds spent ---" % (time.time() - start_time))

  return maps, s1_extend

def predictions_folder(folder_images='/content/drive/My Drive/syntese/data/cnn/images_testing',Folder_models='/content/drive/My Drive/syntese/data/cnn/models/new',Verbose=0):
  '''
    cnn.predictions_folder()

    This function is used to predict fires in many image pairs.
    
    The function takes a path to a folder. In this folder, N image pair folders exists. The function will the find the probability for no fires in each image pair using the cnn.prediction_image() fucntion.
    
    Input:
        folder_images[str]: String of the folder with many image pair folders.
        Folder_models[str]: String of the flder with the many model folders.
        Verbose[int]: Interger deciding the amount of information given. if 0 no info is given. if 1, info will be printed.
        
    Output:
        prediction_maps[list]: List of prediction masks for each image pair.
        extent[list]: Geographic extent of prediction masks. Used for e.g. plotting.
        
    Example:
        maps, extend = predictions_folder(folder_images='/content/drive/My Drive/syntese/data/cnn/images_testing',Folder_models='/content/drive/My Drive/syntese/data/cnn/models/new',Verbose=0)
        

    Author:
      Kristian Soerensen
      July 2020
      kristian.sorensen@hotmail.com
  '''
  #fetching paths of all folders in the folder_images path. [1] is child directories.
  images_paths = next(os.walk(folder_images))[1]
  
  if Verbose>0:
    print(len(images_paths),' Image pairs found')
    
  prediction_maps = []
  extent = []
  #getting prediction for each image in folder using the cnn.prediction_image() function.
  for i in range(len(images_paths)):
    if Verbose>0:
        print('\n -----------------------------\n Getting prediction for image pair: ',str(i+1),'\n -----------------------------')
        
    prediction_mask,ext = prediction_image(folder_img=folder_images+images_paths[i],folder_models=Folder_models,verbose=Verbose)
    extent.append(ext)
    prediction_maps.append(prediction_mask)

  return prediction_maps, extent

def make_dataset(data_folder,split=0.1,verbose=0):
  '''
    A simple function using keras' library to make a dataset. 
    
    The function takes a folder, finds the training data in the folder and makes a training set defined by the split.
    A dataset for Sentinel-1 VV, Sentinel-1 VH, Sentinel-3 S13, Sentinel-3 46 and Sentinel-3 79 are made.
    This function should therefore only be used in this project.
    
    Input:
        data_folder[str]: path of data. The folder structure can be seen in the thesis.
        splot[float]: float between [0,1]. Used to define the amount of validation data from the training data.
        verbose[int]: amount of printed information. 0 nothing is printed. 1 info is printed.
    
    Output:
        s1_vh,s1_vv,s3_13,s3_46,s3_79: dataset for each data tpe.
        
    Example:
        s1_vh,s1_vv,s3_13,s3_46,s3_79 = make_dataset(/home/folder/data_cnn/,split=0.1,verbose=1)
        
    Author:
      Kristian Soerensen
      July 2020
      kristian.sorensen@hotmail.com    

  '''
  if verbose>0:
    print('Making training set')

  s1_vh= []
  s1_vh_val = keras.preprocessing.image_dataset_from_directory(data_folder+'/vh',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='grayscale',seed=1,subset='validation',class_names=['fire','no_fire'],validation_split=split)
  s1_vh_train = keras.preprocessing.image_dataset_from_directory(data_folder+'/vh',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='grayscale',seed=1,subset='training',class_names=['fire','no_fire'],validation_split=split)
  s1_vh.append(s1_vh_val)
  s1_vh.append(s1_vh_train)
  

  s1_vv = []
  s1_vv_val = keras.preprocessing.image_dataset_from_directory(data_folder+'/vv',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='grayscale',seed=1,subset='validation',class_names=['fire','no_fire'],validation_split=split)
  s1_vv_train = keras.preprocessing.image_dataset_from_directory(data_folder+'/vv',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='grayscale',seed=1,subset='training',class_names=['fire','no_fire'],validation_split=split)
  s1_vv.append(s1_vv_val)
  s1_vv.append(s1_vv_train)
  

  s3_13 = []
  s3_13_val = keras.preprocessing.image_dataset_from_directory(data_folder+'/13',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='rgb',seed=1,subset='validation',class_names=['fire','no_fire'],validation_split=split)
  s3_13_train = keras.preprocessing.image_dataset_from_directory(data_folder+'/13',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='rgb',seed=1,subset='training',class_names=['fire','no_fire'],validation_split=split)
  s3_13.append(s3_13_val)
  s3_13.append(s3_13_train)
  


  s3_46 = []
  s3_46_val = keras.preprocessing.image_dataset_from_directory(data_folder+'/46',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='rgb',seed=1,subset='validation',class_names=['fire','no_fire'],validation_split=split)
  s3_46_train = keras.preprocessing.image_dataset_from_directory(data_folder+'/46',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='rgb',seed=1,subset='training',class_names=['fire','no_fire'],validation_split=split)
  s3_46.append(s3_46_val)
  s3_46.append(s3_46_train)
  

  s3_79 = []
  s3_79_val = keras.preprocessing.image_dataset_from_directory(data_folder+'/79',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='rgb',seed=1,subset='validation',class_names=['fire','no_fire'],validation_split=split)
  s3_79_train = keras.preprocessing.image_dataset_from_directory(data_folder+'/79',labels='inferred',label_mode='binary', image_size=(200,200),color_mode='rgb',seed=1,subset='training',class_names=['fire','no_fire'],validation_split=split)
  s3_79.append(s3_79_val)
  s3_79.append(s3_79_train)
  

  return s1_vh,s1_vv,s3_13,s3_46,s3_79


def make_trainig_areas(fire_coord,no_fire_coord,s1_img,s3_img):
    '''
    This function takes a list of coodinates and images and makes subsets.
    Coordiates for two classes are given, then for each class, the corrosponding arrays will be made.
    Here, s1_img and s3_img MUST be of same size.
    
    Input:
        fire_coord[list]=list of coordiates for fire  [row0,row1,colum0,colum1].
        no_fire_coord[list]=list of coordiates for no_fire  [row0,row1,colum0,colum1].
        s1_img[NxMxi]:NxMxi array, where i can be e.g. 2
        s3_img[NxMxj]:NxMxj array, where j can be e.g. 5.
    
    Output:
        fire_box_s1,no_fire_box_s1,fire_box_s3,no_fire_box_s3: Subsets of fire and no fire for each img.
        
    Author:
        Kristian Aalling Soerensen
        June 2020
        kristian.sorensen@hotmail.com  
    '''
    no_fire_box_s1 = []
    no_fire_box_s3 = []
    for i in range(len(no_fire_coord)):
        coord = no_fire_coord[i]
        img_s1 = (np.stack((s1_img[0][coord[0]:coord[1],coord[2]:coord[3]],s1_img[1][coord[0]:coord[1],coord[2]:coord[3]]), axis=2))
        no_fire_box_s1.append(img_s1)
        img_s3 = (np.stack((s3_img[0][coord[0]:coord[1],coord[2]:coord[3]],s3_img[1][coord[0]:coord[1],coord[2]:coord[3]],s3_img[2][coord[0]:coord[1],coord[2]:coord[3]],s3_img[3][coord[0]:coord[1],coord[2]:coord[3]],s3_img[4][coord[0]:coord[1],coord[2]:coord[3]]), axis=2))
        no_fire_box_s3.append(img_s3)
        
    fire_box_s1 = []
    fire_box_s3 = []
    for i in range(len(fire_coord)):
        coord = fire_coord[i]
        img_s1 = (np.stack((s1_img[0][coord[0]:coord[1],coord[2]:coord[3]],s1_img[1][coord[0]:coord[1],coord[2]:coord[3]]), axis=2))
        fire_box_s1.append(img_s1)
        img_s3 = (np.stack((s3_img[0][coord[0]:coord[1],coord[2]:coord[3]],s3_img[1][coord[0]:coord[1],coord[2]:coord[3]],s3_img[2][coord[0]:coord[1],coord[2]:coord[3]],s3_img[3][coord[0]:coord[1],coord[2]:coord[3]],s3_img[4][coord[0]:coord[1],coord[2]:coord[3]]), axis=2))
        fire_box_s3.append(img_s3)
        
    return fire_box_s1,no_fire_box_s1,fire_box_s3,no_fire_box_s3
    

