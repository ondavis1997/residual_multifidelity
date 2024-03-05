from numpy.random import seed
from numpy.random import rand
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(1)
import tensorflow as tf
#import numpy as np
from tensorflow import keras
from keras import optimizers
#from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from numpy import *
from numpy import sin
from numpy import cos 
from numpy import mean
from numpy import linspace
from numpy import argmax

import matplotlib.pyplot as plt
import time

import scipy.io as sio #To save and load mat files for MATLAB

from keras.regularizers import l2
#########################################################################
## 
#########################################################################

#tf.keras.utils.set_random_seed(233)
#tf.keras.utils.set_random_seed(505)
#tf.keras.utils.set_random_seed(500)
tf.keras.utils.set_random_seed(719)


def u_exact(y, t):
    return 0.5 + 2*sin(12*y) + 6*sin(2*t)*sin(10*y)*(1+2*y**2)

def absu_exact(y,t):
    return abs(0.5 + 2*sin(12*y) + 6*sin(2*t)*sin(10*y)*(1+2*y**2))

#def Expectd():
#    E = 0.5*quad(absu_exact, -1,1,epsabs = 1e-14, limit=100, args = (20))[0]
#    return E   

def du(Y, t, u):
    U = u_exact(Y,t)
    g = 12*cos(2*t)*sin(10*Y)*(1+2*Y**2)+0.5*U
    return (- 0.5*u + g)

def initial_u(Y):
    u = u_exact(Y,0)    
    return u

def RK2(f, y, u0, h,T):
    n= int(T/h)+1
    t = 0
    u = u0    
    for i in range(1,n):                        
        k1 = h * f(y, t,u)
        k2 = h * f(y, t + h,u + k1)
        u = u + 0.5*(k1 + k2)
        t = t+ h    
    return u

def RK4(f, y, u0, h,T):
    n= int(T/h)+1
    t = 0
    u = u0    
    for i in range(1,n):                        
        k1 = h * f(y, t,u)
        k2 = h * f(y, t + h/2,u + k1/2)
        k3 = h * f(y, t + h/2,u + k2/2)
        k4 = h * f(y, t + h,u + k3)
        u = u + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t+ h    
    return u

def rel_error(E):
  return abs(E_u_exact - mean(E))/abs(E_u_exact)

def qLHF(Y,T,h):
    U0 = initial_u(Y)
    uLHF = RK2(du, Y, U0, h,T) 
    return uLHF

def my_leaky_relu(xx):
    return tf.nn.leaky_relu(xx, alpha=0.04)
######################################################
#######################################################
    
T=100
E_u_exact = 4.760210172541998 #for T=100
#E_u_exact = 5.467948285412829 #for T=40    


#for tol=0.01
hLF=0.5     
hHF=0.1

#for tol=0.001
#hLF=0.25  
#hHF=0.025   

#for tol=0.0001  
#hLF=0.1      
#hHF=0.01



#hHF=0.05    #for tol=0.005
#hHF=0.02    #for tol=0.0005
#hHF=0.00625 #for tol=0.00005


#######################################
Np = 10**3 
Y=linspace(-1,1,Np+1)

start_time = time.process_time() 
qHF_true=qLHF(Y,T,hHF)
end_time = time.process_time()
CpuTimeHF = (end_time - start_time)/Np

start_time = time.process_time() 
qLF_true=qLHF(Y,T,hLF)
end_time = time.process_time()
CpuTimeLF = (end_time - start_time)/Np


print('W_HF: ', CpuTimeHF)
print('W_LF: ', CpuTimeLF)
#######################################


N=241 #241
yL=linspace(-1,1,N)


Ratio=10
yC=yL[ ::Ratio]                            #takes every Ratio-th entry of yL 
yLnC=delete(yL, slice(None, None, Ratio))  #removes every Ratio-th entry of yL

#qLF=qLHF(yL,T,hLF)
#qLC=qLF[ ::Ratio]


qLnC=qLHF(yLnC,T,hLF)
qLC=qLHF(yC,T,hLF)
qHC=qLHF(yC,T,hHF)


#N=yL.size 
M=yC.size  #M in paper; no. of common points
R=N-M

print('N: ', N)
print('M: ', M)
print('R: ', R)





plt.figure()
plt.figure(dpi=300)
#plt.figure(dpi=1000)
plt.plot(Y,qLF_true,'m-',linewidth=0.3,label='$Q_{LF}$')
plt.plot(Y,qHF_true,'b-',linewidth=0.3,label='$Q_{HF}$')
qLF=qLHF(yL,T,hLF)
plt.plot(yL,qLF,'mo',markersize=1, marker = 'v', fillstyle = 'none', label='available LF data')
plt.plot(yC,qHC,'bo',markersize=2,label='available HF data')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$Q(\theta)$')
plt.legend()
plt.savefig('ODE_train_data.png', format = 'png', dpi = 300, bbox_inches = 'tight')
plt.show()



##############################################################
'''
  SNN to compute QHF-QLF
'''
##############################################################

I=range(1,M-1,20)

yCv=yC[I]
yCt=delete(yC,I)

qLC_scaled=qLC/10
qLCv=qLC_scaled[I]
qLCt=delete(qLC_scaled,I)
x_valid=concatenate((yCv[:,newaxis],qLCv[:,newaxis]),1)
x_train=concatenate((yCt[:,newaxis],qLCt[:,newaxis]),1)

y_SNN=qHC[:,newaxis]-qLC[:,newaxis]
y_valid=y_SNN[I]
y_train=delete(y_SNN,I)


print('DNN x_train shape:', x_train.shape)
print('DNN y_train shape:', y_train.shape)
print('DNN x_valid shape:', x_valid.shape)
print('DNN y_valid shape:', y_valid.shape)



model_shallow = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = x_train.shape[1:]),    
    #tf.keras.layers.Dense(10, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(10, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(10, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),
    tf.keras.layers.Dense(10, activation =my_leaky_relu),
    tf.keras.layers.Dense(10, activation =my_leaky_relu),
    #tf.keras.layers.Dense(5, activation =tf.nn.relu),
    #tf.keras.layers.Dense(5, activation =tf.nn.relu),
    tf.keras.layers.Dense(1,activation=None)
    ])

learn_rate=0.004
no_epoch=100
no_batch=10
start_time = time.process_time() 
#keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
adam_optimizer = tf.keras.optimizers.Adam(lr = learn_rate)
#model_shallow.compile(loss='mean_squared_error', optimizer='adam')
model_shallow.compile(loss='mean_squared_error', optimizer=adam_optimizer)
#model_shallow.compile(loss='mae', optimizer=adam_optimizer)
#history = model_shallow.fit(x_train, y_train, epochs = no_epoch, batch_size = no_batch, verbose = 0)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95,patience=200) 
history = model_shallow.fit(x_train, y_train, epochs = no_epoch, validation_data = (x_valid,y_valid), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr]) 
end_time = time.process_time()
TrainTimeSNN = (end_time - start_time)
print('TrainTime-SNN: ', TrainTimeSNN)

history_dict = history.history
history_dict.keys()
TrainLoss = history_dict['loss']
ValLoss = history_dict['val_loss']
epoch_vec=range(1,len(TrainLoss)+1)

plt.figure()
plt.figure(dpi=300)
plt.plot(epoch_vec,TrainLoss, label = 'TrainLoss')
plt.plot(epoch_vec,ValLoss, label = 'ValLoss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()


train_loss = model_shallow.evaluate(x_train, y_train)     # Evaluate: Returns the loss value for the model in test mode.
print('Train loss:', train_loss)
print('learning rate:', learn_rate)
print('epochs:', no_epoch)
print('batch_size:', no_batch)



################################
#Use SNN to generate the remaining R=N-M HF data
qLnC_scaled=qLnC/10
x_predict=concatenate((yLnC[:,newaxis],qLnC_scaled[:,newaxis]),1)
start_time = time.process_time() 
FnC_predict = model_shallow.predict(x_predict)
qHnC_predict=FnC_predict.reshape(-1)+qLnC
end_time = time.process_time()
PredictTimeSNN = (end_time - start_time)/R




################################
#Stack up all HF data 
x_DNN=hstack((yC,yLnC))
y_DNN=hstack((qHC,qHnC_predict))

plt.figure()
plt.figure(dpi=300)
#plt.figure(dpi=1000)
plt.plot(Y,qHF_true,'b-',linewidth=1,label=r'$Q_{HF}$')
plt.plot(yC,qHC,'bo',markersize=3,label='available HF data')
plt.plot(x_DNN,y_DNN,'rx',markersize=2, marker = 'x', label='generated HF data (by ResNN)')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$Q(\theta)$')
plt.legend()
plt.savefig('ODE_enlarged_train_data.png', format = 'png', dpi = 300, bbox_inches = 'tight')
plt.show()




##############################################################
'''
  DNN to compute QHF
'''
##############################################################

print('x_DNN shape:', x_DNN.shape)
print('y_DNN shape:', y_DNN.shape)


#Valid_Ratio=10  #90% test data and 10% validation data
#x_valid0=x_DNN[ ::Valid_Ratio]                          #takes every Ratio-th entry of x_DNN
#x_train0=delete(x_DNN, slice(None, None, Valid_Ratio))  #removes every Ratio-th entry of x_DNN
#y_valid0=y_DNN[ ::Valid_Ratio]                          #takes every Ratio-th entry of y_DNN
#y_train0=delete(y_DNN, slice(None, None, Valid_Ratio))  #removes every Ratio-th entry of y_DNN

#import random
#I=random.sample(range(N),int(N/10))

I=range(1,N-1,20)
x_valid0=x_DNN[I]
x_train0=delete(x_DNN,I)
y_valid0=y_DNN[I]
y_train0=delete(y_DNN,I)




print('DNN x_train0 shape:', x_train0.shape)
print('DNN y_train0 shape:', y_train0.shape)
print('DNN x_valid0 shape:', x_valid0.shape)
print('DNN y_valid0 shape:', y_valid0.shape)


Neurons40=40
Neurons30=30
Neurons25=25
Neurons20=20
Neurons15=15
Neurons10=10
Neurons5=5

model_deep = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = x_train0.shape[1:]),
    #tf.keras.layers.Dense(15, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(15, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(15, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(15, activation =my_leaky_relu, kernel_regularizer=l2(0.0001)),    
    tf.keras.layers.Dense(20, activation =my_leaky_relu),
    tf.keras.layers.Dense(20, activation =my_leaky_relu),
    tf.keras.layers.Dense(20, activation =my_leaky_relu),
    tf.keras.layers.Dense(20, activation =my_leaky_relu),
    #tf.keras.layers.Dense(Neurons15, activation =tf.nn.relu),
    #tf.keras.layers.Dense(Neurons15, activation =tf.nn.relu),
    #tf.keras.layers.Dense(Neurons15, activation =tf.nn.relu),
    #tf.keras.layers.Dense(Neurons15, activation =tf.nn.relu),
    tf.keras.layers.Dense(1,activation=None)
    ])

learn_rate=0.004 #0.004
no_epoch= 400 #400
no_batch=40
start_time = time.process_time() 
#keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
adam_optimizer = tf.keras.optimizers.Adam(lr = learn_rate)
#model_deep.compile(loss='mean_squared_error', optimizer='adam')
model_deep.compile(loss='mean_squared_error', optimizer=adam_optimizer)
#model_deep.compile(loss='mae', optimizer=adam_optimizer)
#history = model_deep.fit(x_train0, y_train0, epochs = no_epoch, batch_size = no_batch, verbose = 0)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95,patience=300) 
history = model_deep.fit(x_train0, y_train0, epochs = no_epoch, validation_data = (x_valid0,y_valid0), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr]) 
end_time = time.process_time()
TrainTimeDNN = (end_time - start_time)
print('TrainTime-DNN: ', TrainTimeDNN)

history_dict = history.history
history_dict.keys()
TrainLoss = history_dict['loss']
ValLoss = history_dict['val_loss']
epoch_vec=range(1,len(TrainLoss)+1)

plt.figure()
plt.figure(dpi=300)
plt.plot(epoch_vec,TrainLoss, label = 'TrainLoss')
plt.plot(epoch_vec,ValLoss, label = 'ValLoss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
  
  
#plt.plot(epoch_vec, history.history['loss'], label = 'TrainLoss')
#plt.plot(epoch_vec, history.history['val_loss'], label = 'Valloss')


train_loss = model_deep.evaluate(x_train0, y_train0)     # Evaluate: Returns the loss value for the model in test mode.
print('Train loss:', train_loss)
print('learning rate:', learn_rate)
print('epochs:', no_epoch)
print('batch_size:', no_batch)

###################################

#start_time = time.clock() 
qHF_predict = model_deep.predict(Y)
#end_time = time.clock()
#CpuTimeP0 = (end_time - start_time)/Np

plt.figure()
'Ploting some data points'
plt.figure(dpi=300)
#plt.figure(dpi=1000)
plt.plot(Y,qHF_true,'b-',linewidth=1,label=r'$Q_{HF}$')
plt.plot(Y,qHF_predict,'r--',linewidth=1,label=r'$Q_{RMFNN}$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$Q(\theta)$')
plt.legend()
plt.savefig('ODE_RMFNN_prediction.png', format = 'png', dpi = 300, bbox_inches = 'tight')
plt.show()


###################################################
#Save arrays as mat-file for MATLAB
adict = {}
adict['Y'] = Y
adict['qLF_true'] = qLF_true
adict['qHF_true'] = qHF_true
adict['qLF'] = qLF
adict['qHC'] = qHC
adict['qLC'] = qLC
adict['yL'] = yL
adict['yC'] = yC
adict['yLnC'] = yLnC
adict['qHnC_predict']=qHnC_predict
adict['qHF_predict']=qHF_predict
#sio.savemat('Matlab/ODE_01b_matlab.mat', adict)
###################################################
#save files in Python
#savez('ODE_0001_data', Y,qLF_true,qHF_true,qLF,qHC,yL,yC,x_train,y_train,qH_predict)

###################################################

for k in range(1,1+1):
    seed(k+40)
    Nmc = 140*10**3
    YY = 2*rand(Nmc,1)-1
    YY=reshape(YY,Nmc)
            
    start_time = time.process_time() 
    qHF_predict_k = model_deep.predict(YY)
    end_time = time.process_time()
    PredictTimeDNN = (end_time - start_time)/Nmc
    
    
    Error_MCMFNN = rel_error(abs(qHF_predict_k))
    
    print('Total error:', Error_MCMFNN)

#xx=concatenate((YY[:,newaxis],YY[:,newaxis]),1)
#start_time = time.clock() 
#qHF_predict_k = model_shallow.predict(xx)
#end_time = time.clock()
#PredictTimeSNN = (end_time - start_time)/Nmc    
    
print('Predict Time DNN:', PredictTimeDNN)
print('Predict Time SNN:', PredictTimeSNN)

Total_cost=N*CpuTimeLF+M*CpuTimeHF+TrainTimeDNN+TrainTimeSNN+Nmc*PredictTimeDNN+R*PredictTimeSNN
print('Total cost:', Total_cost)




S=100

for k in range(1,1+1):
    #seed(k+530)
    #Nmck = 140*10**7
    Nmck = 10**6
    #YYk = 2*rand(Nmck,1)-1
    
    MSE=0
    Nmc = round(Nmck/S)
    for kk in range(1,1+S):
        #print('kk = ', kk)
        
        #YY=YYk[Nmc*(kk-1):Nmc*kk]
        seed(kk+1)
        YY = 2*rand(Nmc,1)-1
        YY=reshape(YY,Nmc)
        
        qHF_predict_k = model_deep.predict(YY)
        qNN=abs(qHF_predict_k)
        qExact=absu_exact(YY,T)
        MSE=MSE+mean((qExact-qNN.reshape(-1))**2)
        
    MSE=MSE/S
             
    print('MSE error:', MSE)

