#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:11:59 2022

@author: Mashroor Nitol
modified by Daniel N. Blaschke

© 2023. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

Department of Energy/National Nuclear Security Administration. All rights in the program are.

reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

Security Administration. The Government is granted for itself and others acting on its behalf a

nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.

derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.

others to do so.
"""
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
# from keras.models import load_model
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error #, mean_squared_error
from sklearn.preprocessing import StandardScaler #, MinMaxScaler

fntsize = 16

# from tensorflow import keras
fname_csv = 'porosity_velocity_all.csv'
fname_hdf = 'porosity_velocity_all.hdf5'
if os.path.exists(fname_csv):
    dataset = pd.read_csv(fname_csv)
elif os.path.exists(fname_hdf):
    dataset = pd.read_hdf(fname_hdf,'porosity_velocity_all')
else:
    raise FileNotFoundError('cannot find dataset: neither "porosity_velocity_all.csv" nor "porosity_velocity_all.hdf5" exit.')
## check if index was contained in csv file:
if 'Unnamed' in dataset.columns[0] and dataset.columns[1]=='porosity0':
    dataset = dataset.iloc[:,1:]
# dataset = dataset.mask(dataset['TEPLAETA']<0.00027).mask(dataset['VELF']>0.017).dropna()
if 'velocity0' in dataset.columns:
    vel_start_index = dataset.columns.get_loc('velocity0')
elif 'vel_localmax1' in dataset.columns:
    vel_start_index = dataset.columns.get_loc('vel_localmax1')
## uncomment to drop porosity data:
# dataset = dataset.iloc[:,vel_start_index:] ## drop porosity data, keep surface velocity

## restrict to same ranges of tepla parameters as in Thao's data set:
# dataset = dataset.mask(dataset['TEPLAETA']<0.00027).mask(dataset['TEPLAETA']>0.001).\
#     mask(dataset['TEPLAYSPALL']<0.0004).mask(dataset['TEPLAYSPALL']>0.005).\
#     mask(dataset['TEPLAPHI0']<0.0001585).mask(dataset['TEPLAPHI0']>0.00398).dropna()
# dataset = dataset.mask(dataset['TEPLAETA']<1e-3).mask(dataset['TEPLAPHI0']>1e-4).mask(dataset['VELF']<0.02).dropna()
# dataset = dataset.mask(dataset['TFLYER']>0.154).dropna()

X = dataset.iloc[:,:-5].values
y = dataset.iloc[:,-5:].values

in_dim = X.shape[1]
out_dim = y.shape[1]

x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.20)

# reassemble test data into a dataframe and write to csv file for future reference
testdata = pd.concat([pd.DataFrame(x_test),pd.DataFrame(y_test)],axis=1)
testdata.columns = dataset.columns
testdata.to_csv('testdata.csv',index=False)
# remember training data
trainingdata = pd.concat([pd.DataFrame(x_train),pd.DataFrame(y_train)],axis=1)
trainingdata.columns = dataset.columns
trainingdata.to_hdf('trainingdata.hdf5','trainingdata',index=False)
#########

sc_X = StandardScaler()
sc_y = StandardScaler()

x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)
y_test = sc_y.fit_transform(y_test)
y_train = sc_y.fit_transform(y_train)

# model = Sequential()
# model.add(Dense(128, input_dim=in_dim, activation="LeakyReLU",kernel_regularizer='l2'))
# model.add(Dense(64, activation="LeakyReLU",kernel_regularizer='l2'))
# model.add(Dense(128, input_dim=in_dim, activation="relu",kernel_regularizer='l2'))
# model.add(Dense(64, activation="relu",kernel_regularizer='l2'))
# model.add(Dense(128, input_dim=in_dim, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer='l2'))
# model.add(Dense(64, activation=keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer='l2'))
# model.add(Dense(out_dim, activation="linear"))
# model.compile(loss="mse", optimizer="adam",metrics = ['accuracy'])

# model.summary()
# -------------- new model ------------------------------ #
model = Sequential()
model.add(Dense(50, input_dim=in_dim, activation="LeakyReLU",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.add(Dense(200, activation="LeakyReLU",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.add(Dense(out_dim, activation="linear"))
model.compile(loss="mse", optimizer="adam",metrics = ['accuracy'])

epochs = 3000
history = model.fit(x_train, y_train, epochs=epochs, batch_size=10,
          validation_data=(x_test, y_test), verbose=1)

fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
plt.plot(history.history["val_loss"],'b')
plt.plot(history.history["loss"],'r')
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
# plt.axis([0, 1000, 0, 0.2])
plt.ylim(0, 0.25)
# ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
plt.legend(["Test", "Train"], loc="upper right")
# plt.grid()
# plt.text(200,0.1,f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')
plt.savefig('learning_history_5p.pdf',bbox_inches='tight')
plt.show()

# Get the predicted values
y_pred = model.predict(x_test)
y_oI = sc_y.inverse_transform(y_test)
y_pI = sc_y.inverse_transform(y_pred)

# y_oI = y_test
# y_pI = model.predict(x_test)

# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_pI, y_oI)
print(f'Median Absolute Error (MAE): {np.round(MAE, 6)}')

# # Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_oI, y_pI)/ y_oI))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# # Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_oI, y_pI)/ y_oI)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

model.save("model.h5")

fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
plt.plot(history.history["val_loss"],'b')
plt.plot(history.history["loss"],'r')
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
# plt.axis([0, 1000, 0, 0.2])
plt.ylim(0, 0.25)
# ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
plt.legend(["Test", "Train"], loc="upper right")
# plt.grid()
plt.text(250,0.19,f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')
plt.savefig('learning_history_5p_wl.pdf',bbox_inches='tight')

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, constrained_layout=False, figsize=(24,4))
errors = (np.abs(np.subtract(y_oI, y_pI)/ y_oI))

ax1.set_title(r'$Y_\mathrm{S}$',fontsize=fntsize)
ax1.plot(y_oI[:,4],y_pI[:,4],'ro',alpha = 0.5,label=f'{np.mean(errors[:,-1])*100:.2f} % mean error')
ax1.plot(y_oI[:,4],y_oI[:,4],'r-')
ax1.ticklabel_format(style='sci',scilimits=(0,0))
ax1.legend(loc='upper left',handlelength=0., frameon=False, shadow=False,fontsize=fntsize)
ax1.xaxis.set_tick_params(labelsize=fntsize)
ax1.yaxis.set_tick_params(labelsize=fntsize)
ax1.xaxis.offsetText.set_fontsize(fntsize)
ax1.yaxis.offsetText.set_fontsize(fntsize)

ax2.set_title(r'$\phi_{0}$',fontsize=fntsize)
ax2.plot(y_oI[:,3],y_pI[:,3],'bo',alpha = 0.5,label=f'{np.mean(errors[:,-2])*100:.2f} % mean error')
ax2.plot(y_oI[:,3],y_oI[:,3],'b-')
ax2.ticklabel_format(style='sci',scilimits=(0,0))
ax2.legend(loc='upper left',handlelength=0., frameon=False, shadow=False,fontsize=fntsize)
ax2.xaxis.set_tick_params(labelsize=fntsize)
ax2.yaxis.set_tick_params(labelsize=fntsize)
ax2.xaxis.offsetText.set_fontsize(fntsize)
ax2.yaxis.offsetText.set_fontsize(fntsize)

ax3.set_title(r'$\eta$',fontsize=fntsize)
ax3.plot(y_oI[:,2],y_pI[:,2],'ko',alpha = 0.5,label=f'{np.mean(errors[:,-3])*100:.2f} % mean error')
ax3.plot(y_oI[:,2],y_oI[:,2],'k-')
ax3.ticklabel_format(style='sci',scilimits=(0,0))
ax3.legend(loc='upper left',handlelength=0., frameon=False, shadow=False,fontsize=fntsize)
ax3.xaxis.set_tick_params(labelsize=fntsize)
ax3.yaxis.set_tick_params(labelsize=fntsize)
ax3.xaxis.offsetText.set_fontsize(fntsize)
ax3.yaxis.offsetText.set_fontsize(fntsize)

ax4.set_title(r'$v$',fontsize=fntsize)
ax4.plot(y_oI[:,1],y_pI[:,1],'go',alpha = 0.5,label=f'{np.mean(errors[:,-4])*100:.2f} % mean error')
ax4.plot(y_oI[:,1],y_oI[:,1],'g-')
ax4.ticklabel_format(style='sci',scilimits=(0,0))
ax4.legend(loc='upper left',handlelength=0., frameon=False, shadow=False,fontsize=fntsize)
ax4.xaxis.set_tick_params(labelsize=fntsize)
ax4.yaxis.set_tick_params(labelsize=fntsize)
ax4.xaxis.offsetText.set_fontsize(fntsize)
ax4.yaxis.offsetText.set_fontsize(fntsize)

ax5.set_title(r'TFLYER',fontsize=fntsize)
ax5.plot(y_oI[:,0],y_pI[:,0],'mo',alpha = 0.5,label=f'{np.mean(errors[:,-5])*100:.2f} % mean error')
ax5.plot(y_oI[:,0],y_oI[:,0],'m-')
ax5.ticklabel_format(style='sci',scilimits=(0,0))
ax5.legend(loc='upper left',handlelength=0., frameon=False, shadow=False,fontsize=fntsize)
ax5.xaxis.set_tick_params(labelsize=fntsize)
ax5.yaxis.set_tick_params(labelsize=fntsize)
ax5.xaxis.offsetText.set_fontsize(fntsize)
ax5.yaxis.offsetText.set_fontsize(fntsize)

fig.text(0.5, -0.05, 'Actual value', ha='center',fontsize=fntsize)
fig.text(0.1, 0.5, 'Deep NN prediction', va='center', rotation='vertical',fontsize=fntsize)

plt.savefig('tepla_prediction_deepnn.pdf', bbox_inches='tight')
plt.close()


# write predicted parameters of test data to csv file
predicted_data = pd.DataFrame(y_pI)
predicted_data.columns = dataset.columns[-5:]
predicted_data.to_csv('predicted_data.csv',index=False)
