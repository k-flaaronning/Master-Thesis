# -*- coding: utf-8 -*-
"""Imputation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ykiKbUBPB8LIdvRPY9NptpatSIK6znHb
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

!kill -9 -1



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

downloaded = drive.CreateFile({'id':'1nSAqJYO5thSWS0gFWDgmTTtsN7DJtQKD'}) # replace the id with id of file you want to access
downloaded.GetContentFile('Testing_Set1.csv')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import random as random 
from random import choice 
import numpy as np 
import pandas as pd 
import sys 
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing
from sklearn.impute import (SimpleImputer, KNNImputer, IterativeImputer)
import random as random 
from random import choice 
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import (SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator)
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
import random as random 
from random import choice

#Loading the DataFrame
df = pd.read_csv('Testing_Set1.csv', sep = ';') 
df = df.iloc[1500:5000]
df_nan = pd.DataFrame(new_dataframe(df, 0.025))

#Randomly removing values from the original dataset
def new_dataframe(data, percent):
    data = data.to_numpy()
    mat = data.copy()
    prop = int(mat.size * percent)
    mask = random.sample(range(mat.size), prop)
    np.put(mat, mask, [np.NaN]*len(mask))

    return mat

#Calculating the Root Mean Square Error and Standard Deviation for the imputed and the original value
def mean_error(original, predicted):
  original = original.to_numpy()
  output = []
  
  for i in range(len(original)):
    x = np.sqrt(((original[i] - np.asscalar(predicted[i]))**2))
    output.append(x)

  output = np.array(output)
  return output

#Univariate Imputation Techinques 
from sklearn.metrics import mean_squared_error
data_median = pd.DataFrame(data_nan.fillna(data_nan.median()))
data_mean = pd.DataFrame(data_nan.fillna(data_nan.mean()))
data_ff = pd.DataFrame(data_nan.fillna(method = 'ffill'))
data_bf = pd.DataFrame(data_nan.fillna(method = 'bfill'))
data_linear = data_nan.interpolate(method = 'linear', axis = 0).ffill().bfill()
data_poly = data_nan.interpolate(method = 'polynomial', order = 5, axis = 0).ffill().bfill()

#Calculation of RMSE and STD
error_median = mean_error(df['HP Flare'], data_median)
error_mean = mean_error(df['HP Flare'], data_mean)
error_ff = mean_error(df['HP Flare'], data_ff)
error_bf = mean_error(df['HP Flare'], data_bf)
error_linear = mean_error(df['HP Flare'], data_linear)
error_poly = mean_error(df['HP Flare'], data_poly)

#Imputation for the variabel HP-Flare using several univariate techniques
data_nan = pd.DataFrame(df_nan[69])
df_1 = pd.DataFrame()
df_1['Nan'] = df_nan[69]
df_1['Mean'] = pd.DataFrame(data_nan.fillna(data_nan.mean()))
df_1['median'] = pd.DataFrame(data_nan.fillna(data_nan.median()))

df_1['Foward'] = data_nan.fillna(method = 'ffill')
df_1['BackWard'] = data_nan.fillna(method = 'bfill')
df_1['Linear'] = data_nan.interpolate(method = 'linear', axis = 0).ffill().bfill()
df_1['Poly']= data_nan.interpolate(method = 'polynomial', order = 5, axis = 0).ffill().bfill()

#Using KNN-Iterative Impuation 
data_knn = df_nan.copy()
imputer_knn = IterativeImputer(estimator = KNeighborsRegressor(n_neighbors = 10), random_state = 0, max_iter = 1000, tol = 0.08)
imputer_knn.fit(data_knn)
data_knn = imputer_knn.transform(data_knn)



# Using KNN Simple Imputer
data_knn1 = df_nan.copy()
imputer_knn1 = KNNImputer()
imputer_knn1.fit(data_knn1)
data_knn1 = imputer_knn1.transform(data_knn1)

#Using ExtraTreeRegressor Iterative Imputer
data_exr = df_nan.copy()
imputation_exr = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), missing_values=np.nan, sample_posterior=False, 
                                 max_iter=1000, tol=0.1, 
                                 n_nearest_features=4, initial_strategy='median')

imputation_exr.fit(data_exr)
data_exr = imputation_exr.transform(data_exr)

# Iterative Decision Tree Regressor Imputation:
data_dt = df_nan.copy()
imputer_dt = IterativeImputer(estimator = DecisionTreeRegressor(), missing_values = np.nan, sample_posterior = False, 
                             max_iter = 1000, tol = 0.1, n_nearest_features = 4, imputation_order = 'ascending')
imputer_dt.fit(data_dt)
data_dt = imputer_dt.transform(data_dt)

# Iterative Bayesian Ridge Regressor
data_bayesian = df_nan.copy()
imputer_br = IterativeImputer(estimator = BayesianRidge(), missing_values = np.nan, random_state = 0, n_nearest_features = 5, sample_posterior = True)
imputer_br.fit(data_bayesian)
data_bayesian = imputer_br.transform(data_bayesian)

#RMSE AND STD FOR ITERATIVE IMPUTER
def mean_error_iterative(original, predicted):
  original = original.to_numpy()
  predicted_df = pd.DataFrame(predicted)
  predicted_feature = predicted_df[69]
  output = []
  
  for i in range(len(original)):
    x = np.sqrt(((original[i] - np.asscalar(predicted_feature[i]))**2))
    output.append(x)

  output = np.array(output)
  return output

error_iterativeknn = mean_error_iterative(df['HP Flare'], data_knn)
error_simpleknn = mean_error_iterative(df['HP Flare'], data_knn1)
error_extratreeregressor = mean_error_iterative(df['HP Flare'], data_exr)
error_decisiontree = mean_error_iterative(df['HP Flare'], data_dt)
error_bayesian = mean_error_iterative(df['HP Flare'], data_bayesian)

print( 'ITERATIVE KNN:', error_iterativeknn.mean(), error_iterativeknn.std())
print( 'SIMPLE KNN:', error_simpleknn.mean(), error_simpleknn.std())
print( 'EXTRA TREE REGRESSOR:', error_extratreeregressor.mean(), error_extratreeregressor.std())
print( 'DECISION TREE:', error_decisiontree.mean(), error_decisiontree.std())

# Looking at the HP FLARE Variabel
nan = pd.DataFrame(df_nan)
res_knn = pd.DataFrame(data_knn)
res_knn1 = pd.DataFrame(data_knn1)
res_exr = pd.DataFrame(data_exr)
res_dec = pd.DataFrame(data_dt)
res_bayesian = pd.DataFrame(data_bayesian)
res = pd.DataFrame()
res['NaN'] = nan[69]
res['Iterative KNN'] = res_knn[69]
res['KNN SimpleImputer'] = res_knn1[69]
res['Iterative Extra Tree Regressor'] = res_exr[69]
res['Iterative Decision Tree'] = res_dec[69]
res['Iterative Bayesian Ridge'] = res_bayesian[69]

res.to_excel("Multivariate_HPFLARE.xlsx") 
df['HP Flare'].to_excel('Original.xlsx')

# Batch Generator for the RNN Network
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

"""Tentative Model for Validation the Imputation Techniques"""

split = [0.5,0.6,0.7,0.8,0.9]

from sklearn.metrics import mean_squared_error
#split = [0.8]
def run_model(data, df_targets):
  #final_mse = np.empty((len(split)))
  final_mse = []
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

    #model_mse = np.empty((2))
    model_mse = []
    for i in range(2):
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
                epochs=10,
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

      temp_mse = np.sqrt(mean_squared_error(y_test, y_pred_rescaled)) 
      temp_mse = temp_mse.item()
      #print(temp_mse)
      model_mse.append(temp_mse)
      #model_mse = np.append(model_mse, temp_mse)
      #np.insert(model_mse,0, temp_mse)
      print('model finished')


    print('split finished')
    #print(model_mse)
    #print(np.mean(model_mse))
    final_mse.append(np.mean(temp_mse))
    #final_mse = np.append(final_mse, np.mean(model_mse))
    #final_mse.insert(0, np.mean(model_mse))
    #np.insert(final_mse, 0, model_mse)

  return_final_mse = np.array(final_mse)



  return return_final_mse

"""First Imputing the missing values, later predict the value of HP Flare with the use of a RNN"""

#Imputation with Linear Interpolation
data_interpolation = new.copy()
data_interpolation = data_interpolation.interpolate(method = "linear", axis = 0).ffill().bfill()
interpolation_score = run_model(data_interpolation, df_targets)

#Imputation with Fill Forward
data_ff = new.copy()
data_ff = data_ff.fillna(method = "ffill")
data_ff = data_ff.fillna(method = "bfill")
ff_score = run_model(data_ff, df_targets)

#Imputation with BackWard Fill
data_bf = new.copy()
data_bf = data_bf.fillna(method = "bfill")
data_bf = data_bf.fillna(method = "ffill")
bf_score = run_model(data_bf, df_targets)

#Imputation with Simple KNN Imputer
data_knn = new.copy()
imputer_knn = KNNImputer()
imputer_knn.fit(data_knn)
data_knn = imputer_knn.transform(data_knn)
data_knn = pd.DataFrame(data_knn)
result_knn = run_model(data_knn, df_targets)

#Imputation with KNN Iterative Imputer
data_knn1 = new.copy()
knniterative_imputer = IterativeImputer(estimator = KNeighborsRegressor(n_neighbors = 10), random_state = 0, max_iter = 1000, tol = 0.08)
knniterative_imputer = knniterative_imputer.fit(data_knn1)
data_knn1 = knniterative_imputer.transform(data_knn1)
data_knn1 = pd.DataFrame(data_knn1)
result_iterativeknn = run_model(data_knn1, df_targets)

#Iterative DecisionTree()
data_decision = new.copy()
decisiontree_imputer = IterativeImputer(estimator = DecisionTreeRegressor(max_features = 'sqrt', random_state = 0), random_state = 0, max_iter = 1000, tol = 0.1)
decisiontree_imputer = decisiontree_imputer.fit(data_decision)
data_decision = decisiontree_imputer.transform(data_decision)
data_decision = pd.DataFrame(data_decision)
result_decison = run_model(data_decision, df_targets)

results = ( (interpolation_score.mean(), interpolation_score.std()), 
           (ff_score.mean(), ff_score.std()), (bf_score.mean(), bf_score.std()), (result_knn.mean(), result_knn.std()), 
           (result_iterativeknn.mean(), result_iterativeknn.std()), 
            (result_decison.mean(), result_decison.std()))

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