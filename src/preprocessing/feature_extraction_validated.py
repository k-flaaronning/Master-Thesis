# -*- coding: utf-8 -*-
"""Feature_Extraction-Validated.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f1XU3pq9NsYx06AlDR_0jpQQc7qaQeO1
"""

import tensorflow as tf
tf.test.gpu_device_name()

# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

#!kill -9 -1

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
  print('re-execute this cell.')
else:
  print('You are using a high-RAM runtime!')

# Code to read csv file into colaboratory:
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

downloaded = drive.CreateFile({'id':'1BICvjOtNMoAgB6kj6JJgzC2KHtlB4d1Q'}) # replace the id with id of file you want to access
downloaded.GetContentFile('Data_namechanged.pkl')

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import pandas.util.testing as tm

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from tensorflow.keras.losses import MeanSquaredError

def batch_generator(batch_size, sequence_length, num_x_signals, num_y_signals, num_train, x_train_scaled, y_train_scaled):
   
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        for i in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)
            
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

warmup_steps = 50
def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    mse = mean(square(y_true_slice - y_pred_slice))
    
    return mse

def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

data = pd.read_pickle('Data_namechanged.pkl')

# data = data[['Discharge Pressure GTA', 'Suction Gas Temprature CC3', 'Suction Gas Pressure CA2', 
#                    'Discharge Pressure GTB', 'Suction Gas Pressure CB2', 'Suction Gas Temprature CA3',
#                    'Suction Gas Temperature CA1', 'Suction Gas Pressure CC2','Discharge Gas Pressure CC3',
#                    'Suction Gas Temprature CB3','Suction Gas Temperature CC1','Suction Gas Temperature CB1',
#                    'OPRA3 Gas Temperature','OPRA2 Gas Temperature','Deg Heading','Wind Direction 2 ',
#                    'Suction Gas Pressure CC1','Air Inlet Temperature - GTA','Wind Speed 1','Wind Direction 1 ',
#                    'Wind Speed  2','OPRA1 Gas Temperature', 'HP Flare']]
data.head()
data = data.interpolate(method='linear') 
#shift_steps = 5
df_targets = data.pop('HP Flare')#.shift(-shift_steps)



split = [0.5,0.6,0.7,0.8,0.9]
#split = [0.8]
from sklearn.metrics import mean_squared_error

def run_model(data):
  #final_mse = np.empty((len(split)))
  final_rmse = []
  for temp_split in split:
    x_data = data.values
    y_data = df_targets.values.reshape(-1,1)

    num_data = len(x_data)
    train_split = temp_split
    num_train = int(train_split * num_data)
    num_test = num_data - num_train

    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]

    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    batch_size = 256
    sequence_length = 100

    generator = batch_generator(batch_size, sequence_length, num_x_signals, num_y_signals, num_train, x_train_scaled, y_train_scaled)

    validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))

    model = Sequential()
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(None, num_x_signals,)))
    model.add(Dense(num_y_signals, activation='sigmoid'))

    if False:
        from tensorflow.python.keras.initializers import RandomUniform

        # Maybe use lower init-ranges.
        init = RandomUniform(minval=-0.05, maxval=0.05)

        model.add(Dense(num_y_signals,
                      activation='linear',
                        kernel_initializer=init))

    optimizer = RMSprop(lr=1e-3)
    model.compile(loss=loss_mse_warmup, optimizer=optimizer, metrics = ['mse'])

    path_checkpoint = 'best_model'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks = [callback_checkpoint, callback_early_stopping]

    model.fit(x=generator,
              epochs=1,
              steps_per_epoch=100,
              validation_data=validation_data, 
              callbacks = callbacks)
    try:
      model.load_weights(path_checkpoint)
      print('Success')
    except Exception as error:
      print("Error trying to load checkpoint.")
      print(error)

    # Input-signals for the model.
    x = np.expand_dims(x_test_scaled, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    out_rmse = np.sqrt(mean_squared_error(y_test,y_pred_rescaled))
    
    print(out_rmse)
    #final_mse = np.append(final_mse, out_mse)
    final_rmse.append(out_rmse)

  return_final_rmse = np.array(final_rmse)



  return return_final_rmse

"""Testing different feature extraction mechanisms:"""

from sklearn.decomposition import PCA

def Feature_Extraction(data):

  #With the Full Dataset
  full_score = run_model(data)

  # PCA preserving 95 percent of the variance
  data_pc = data.copy()
  pca_95 = PCA(n_components = 0.95)
  projected_95 = pca_95.fit_transform(data_pc)
  projected_95 = pd.DataFrame(projected_95)
  output_95 = run_model(projected_95)

  # PCA preserving 99 percent of the variance
  data_pc2 = data.copy()
  pca_99 = PCA(n_components = 0.99)
  projected_99 = pca_99.fit_transform(data_pc2)
  projected_99 = pd.DataFrame(projected_99)
  output_99 = run_model(projected_99)

  # PCA preserving 999 percent of the variance
  data_pc3 = data.copy()
  pca_999 = PCA(n_components = 0.999)
  projected_999 = pca_999.fit_transform(data_pc3)
  projected_999 = pd.DataFrame(projected_999)
  output_999 = run_model(projected_999)

  kpca_n3 = data.copy()
  
  
  results = ((full_score.mean(), full_score.std()), (output_95.mean(), output_95.std()), 
    (output_99.mean(), output_99.std()), (output_999.mean(), output_999.std()))

  results = np.array(results)
  names = ['Full Data', 'PCA 3 Dimension', 'PCA 5 Dimensions', 'PCA 9 Dimensions']

  return results, names

result = Feature_Extraction(data)

import matplotlib.pyplot as plt

def plot_function(output):
  results = np.array(output[0])
  names = output[1]
  mse = results[:,0]
  std = results[:,1]

  bars = len(mse)
  xval  = np.arange(bars)

  colors = ['Lightblue']
  plt.figure(figsize=(12,6))
  ax1 = plt.subplot(121)

  for i in xval:
    ax1.barh(i, mse[i], xerr = std[i], color = 'Lightblue', alpha = 0.6, align = 'center')
    ax1.set_xlim(left = np.min(mse)*0.8, right = np.max(mse)*1.2)
    ax1.set_yticks(xval)
    ax1.set_xlabel('Root Mean Squared Error w/ Standard Deviation')
    ax1.invert_yaxis()
    ax1.set_yticklabels(names)

  plt.show()

plot_function(result)
