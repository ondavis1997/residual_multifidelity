from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import tensorflow as tf
#import numpy as np
from tensorflow import keras
from keras import optimizers
#from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from numpy import *
import matplotlib.pyplot as plt
import time

import scipy.io as sio #To save mat files for MATLAB
import numpy as np


from keras.regularizers import l2

#np.set_printoptions(precision = 2)
#########################################################################
## 
#########################################################################


def u_exact(t,x,y,w,kx,ky):
    a=np.sin(w*t-kx*x)
    b=np.sin(ky*y)
    return np.multiply(a,b)


def f2_fun(x,y,w,kx,ky):
    a=np.cos(kx*x)
    b=np.sin(ky*y)
    return w*np.multiply(a,b)

def forcing(t,x,y,w,kx,ky):
    a=np.sin(w*t-kx*x)
    b=np.sin(ky*y)
    return -(w**2-kx**2-ky**2)*np.multiply(a,b)

def laplac(u,hx,hy,nx,ny):
    L= np.zeros((ny,nx))
    L[1:-1,1:-1]=(1/hx**2)*u[1:-1,2:]+(1/hx**2)*u[1:-1,0:-2]+(1/hy**2)*u[2:,1:-1]+(1/hy**2)*u[0:-2,1:-1]-((2/hx**2)+(2/hy**2))*u[1:-1,1:-1]
    return L

#####################################################
def  wave_solver(T,xQ,yQ,nx,w,kx,ky):
    #w=10 # \in [10,11]; 
    #kx=6 
    #ky=4
    
    #QoI: Q=u(T,xQ,xQ)
    #xQ=0.5
    
    #nx = 101                                                #number of grid points in x direction
    ny=nx                                                  #equal number of grid points in x and y directions
    side_length = 2                                        #Length of one side of computational domain
    hx, hy = side_length/(nx-1), side_length/(ny-1)        #grid size
    x = np.linspace(-1,1,nx)                               #Spatial grid points
    y = np.linspace(-1,1,nx) 
    X, Y = np.meshgrid(x, y)
    #c = 1.0                                                #Constant speed in the 2D wave equation.
    
    #Set up the time grid to calcuate the equation.
    #T =  20                                                 #Final time (s)
    dt = 0.5*hx                                             #Time step size to ensure a stable discretization scheme.
    nt = int(T/dt)                                          #Total number of time steps.
    dt=T/nt
    
    
    #Set up initial data and obtain u0 and u1
    u0=u_exact(0,X,Y,w,kx,ky)
    f2=f2_fun(X,Y,w,kx,ky)
    L=laplac(u0,hx,hy,nx,ny)
    f=forcing(0,X,Y,w,kx,ky)
    u1=u0+dt*f2+0.5*(dt**2)*(L+f)
    
    #update BC
    u1[0,:]=u_exact(dt,x,-1,w,kx,ky)
    u1[-1,:]=u_exact(dt,x,1,w,kx,ky)
    u1[:,0]=u_exact(dt,-1,y,w,kx,ky)
    u1[:,-1]=u_exact(dt,1,y,w,kx,ky)
    
    
    for k in range(0,nt-1):
        t=(k+1)*dt
        #obtin right hand side
        f=forcing(t,X,Y,w,kx,ky)
        L=laplac(u1,hx,hy,nx,ny)
        #March in time
        u2=2*u1-u0+(dt**2)*(L+f)
        #update BC
        u2[0,:]=u_exact(t+dt,x,-1,w,kx,ky)
        u2[-1,:]=u_exact(t+dt,x,1,w,kx,ky)
        u2[:,0]=u_exact(t+dt,-1,y,w,kx,ky)
        u2[:,-1]=u_exact(t+dt,1,y,w,kx,ky)
        #switch solution at different time levels
        u0=u1
        u1=u2  
        
    mQ=int(round(1+0.5*(nx-1)*(1+xQ)))
    #print(mQ)
    Q=u2[mQ-1,mQ-1]
    #Qexact=u_exact(T,xQ,xQ,w,kx,ky)
    #print('Qpredicted:', Q)
    #print('Qexact:', Qexact)
            
    #fig = plt.figure()
    #ax = fig.gca(projection = '3d')
    #surf = (ax.plot_surface(X,Y,u2, rstride=2, cstride=2,
    #                    cmap ='coolwarm', vmax = 1, vmin = -1, linewidth=1))
    #ax.set_title('2D Wave QoI')
    #plt.show()
    return Q

#######################################################
#######################################################
#######################################################
T=30
xQ=0.5
EQ_exact=0.358525224

#for tol=0.1
nLF=41
nHF=65
    
#for tol=0.01
#nLF=65
#nHF=257     

#for tol=0.001
#nLF=65
#nHF=641    

#######################################
#Step 1


#for tol=0.1
Mw=106
Mk=33

#for tol=0.01
#Mw=121
#Mk=41

#for tol=0.001
#Mw=151
#Mk=51


W1d=linspace(10,11,Mw)
K1d=linspace(4,6,Mk)
W, K = np.meshgrid(W1d, K1d)
RatioW=3
RatioK=4


W1=W[ ::RatioK, ::RatioW]                   
W2a=delete(W, list(range(0, W.shape[0], RatioK)), axis=0)
W2b=W[ ::RatioK,:] 
W2b=delete(W2b, list(range(0, W.shape[1], RatioW)), axis=1)

K1=K[ ::RatioK, ::RatioW]                   
K2a=delete(K, list(range(0, K.shape[0], RatioK)), axis=0)
K2b=K[ ::RatioK,:] 
K2b=delete(K2b, list(range(0, K.shape[1], RatioW)), axis=1)


M=Mw*Mk
M1w=W1.shape[1] 
M1k=W1.shape[0]
M1=M1w*M1k   #M1 in paper; no. of common points M1=W1.size
M2=M-M1      #M2=W2a.size + W2b.size

W1=reshape(W1,(M1,))
K1=reshape(K1,(M1,))
W2a=reshape(W2a,(W2a.size,))
W2b=reshape(W2b,(W2b.size,))
K2a=reshape(K2a,(K2a.size,))
K2b=reshape(K2b,(K2b.size,))
W2=hstack((W2a,W2b))
K2=hstack((K2a,K2b))

print('M=M1+M2:', M)
print('M1:', M1)
print('M2:', M2)
#######################################
#Step 2

qHF1=np.zeros((M1,))
start_time = time.clock() 
for i in range(0,M1):
    #print(i)
    #qHF1[i]=wave_solver(T,xQ,xQ,nHF,W1[i],K1[i],K1[i])
    qHF1[i]=u_exact(T,xQ,xQ,W1[i],K1[i],K1[i])

end_time = time.clock()
CpuTimeHF = (end_time - start_time)/M1

qLF1=np.zeros((M1,))
for i in range(0,M1): 
    print(i)
    qLF1[i]=wave_solver(T,xQ,xQ,nLF,W1[i],K1[i],K1[i])

qLF2=np.zeros((M2,))   
start_time = time.clock()  
for i in range(0,M2):
    print(i)
    qLF2[i]=wave_solver(T,xQ,xQ,nLF,W2[i],K2[i],K2[i])

end_time = time.clock()
CpuTimeLF = (end_time - start_time)/M2


print('CPpuTimeLF:', CpuTimeLF)
print('CPpuTimeHF:', CpuTimeHF)

#fig = plt.figure()
#plt.figure(dpi=100)
#ax = fig.gca(projection = '3d')
#ax.scatter(W2, K2, qLF2)
#plt.show()


###############################################################
#Step 3

##############################################################
'''
  SNN to compute QHF-QLF = F(y, qL(y)) where y=(w,k)
'''
##############################################################

Valid_Ratio=20  #90% test data and 10% validation data
W1v=W1[ ::Valid_Ratio]                          #takes every Ratio-th entry of W1 
W1t=delete(W1, slice(None, None, Valid_Ratio))  #removes every Ratio-th entry of W1
K1v=K1[ ::Valid_Ratio]
K1t=delete(K1, slice(None, None, Valid_Ratio))
qLF1v=qLF1[ ::Valid_Ratio]
qLF1t=delete(qLF1, slice(None, None, Valid_Ratio))
qHF1v=qHF1[ ::Valid_Ratio]
qHF1t=delete(qHF1, slice(None, None, Valid_Ratio))

x_train=concatenate((W1t[:,newaxis],K1t[:,newaxis],qLF1t[:,newaxis]),1)
y_train=qHF1t[:,newaxis]-qLF1t[:,newaxis]

x_valid=concatenate((W1v[:,newaxis],K1v[:,newaxis],qLF1v[:,newaxis]),1)
y_valid=qHF1v[:,newaxis]-qLF1v[:,newaxis]

print('NN1 x_train shape (M1T x 3):', x_train.shape)
print('NN1 y_train shape:', y_train.shape)
print('NN1 x_valid shape (M1V x 3):', x_valid.shape)
print('NN1 y_valid shape:', y_valid.shape)


Neurons50=50
Neurons40=40
Neurons30=30
Neurons28=28
Neurons20=20
Neurons10=10

model_shallow = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = x_train.shape[1:]),    
    tf.keras.layers.Dense(Neurons20, activation =tf.nn.relu, kernel_regularizer=l2(0.0001)),
    tf.keras.layers.Dense(Neurons20, activation =tf.nn.relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(Neurons30, activation =tf.nn.relu, kernel_regularizer=l2(0.0001)),
    #tf.keras.layers.Dense(Neurons30, activation =tf.nn.relu, kernel_regularizer=l2(0.0001)),
    tf.keras.layers.Dense(1,activation=None)
    ])

learn_rate=0.005
no_epoch=100
no_batch=50
start_time = time.clock() 
#keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
adam_optimizer = tf.keras.optimizers.Adam(lr = learn_rate)
#model_shallow.compile(loss='mean_squared_error', optimizer='adam')
#model_shallow.compile(loss='mae', optimizer=adam_optimizer)
model_shallow.compile(loss='mean_squared_error', optimizer=adam_optimizer)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.98,patience=200) 
#history = model_shallow.fit(x_train, y_train, epochs = no_epoch, validation_split = 0.2, shuffle = True, batch_size = no_batch, verbose = 0, callbacks=[reduce_lr])  
history = model_shallow.fit(x_train, y_train, epochs = no_epoch, validation_data = (x_valid,y_valid), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr]) 
end_time = time.clock()
CpuTimeT1 = (end_time - start_time)
print('SNN Train time:', CpuTimeT1)

history_dict = history.history
history_dict.keys()
TrainLoss = history_dict['loss']
ValLoss = history_dict['val_loss']
epoch_vec=range(1,len(TrainLoss)+1)
plt.figure(dpi=100)
plt.plot(epoch_vec,TrainLoss, label = 'Train-Loss')
plt.plot(epoch_vec,ValLoss, label = 'Val-Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

train_loss = model_shallow.evaluate(x_train, y_train)     # Evaluate: Returns the loss value for the model in test mode.
print('Train loss:', train_loss)
print('Val Loss:', ValLoss[-1])
print('learning rate:', learn_rate)
print('epochs:', no_epoch)
print('batch_size:', no_batch)

#test_loss = model_shallow.evaluate(x_test, y_test)     # Evaluate: Returns the loss value for the model in test mode.
# print('Test loss:', test_loss)



################################
#Use SNN to generate the remaining M2=M-M1 HF data

x_predict=concatenate((W2[:,newaxis],K2[:,newaxis],qLF2[:,newaxis]),1)
start_time = time.clock() 

FnC_predict = model_shallow.predict(x_predict)
end_time = time.clock()
qHF2_predict=FnC_predict.reshape(-1)+qLF2

PredictTimeSNN = (end_time - start_time)/M2





qHF2=np.zeros((M2,1)) 
for i in range(0,M2):
#    qHF2[i]=wave_solver(T,xQ,xQ,nHF,W2[i],K2[i],K2[i])
    qHF2[i]=u_exact(T,xQ,xQ,W2[i],K2[i],K2[i])

#plt.figure()
#plt.plot(qHF2,'b-',label='$q_{HF}$')
#plt.plot(qHF2_predict,'r-',label='$q_{NN1}$')
#plt.legend()
#plt.show()

print('MAX Error', max(abs(qHF2-qHF2_predict[:,newaxis])))
print('MEAN Error', mean(abs(qHF2-qHF2_predict[:,newaxis])))

adict = {}
adict['W1'] = W1
adict['K1'] = K1
adict['W2'] = W2
adict['K2'] = K2
adict['qLF1'] = qLF1
adict['qHF1'] = qHF1
adict['qHF2'] = qHF2
adict['qHF2_predict'] = qHF2_predict
#sio.savemat('PDE_1a_matlab.mat', adict)






#if False: '''

###################################################
#Step 4

##############################################################
'''
  DNN to compute QHF
'''
##############################################################


Wnn2=hstack((W1,W2))-10.0
Knn2=hstack((K1,K2))*0.5-2.0
qHFnn2=hstack((qHF1,qHF2_predict.reshape(-1)))
#Wnn2=delete(Wnn2, slice(None, None, 2))
#Knn2=delete(Knn2, slice(None, None, 2))
#qHFnn2=delete(qHFnn2, slice(None, None, 2))


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(W1, K1, qHF1, c=qHF1, cmap='viridis', linewidth=0.5);
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(W1, K1, qHF1, cmap='viridis', edgecolor='none');
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(W2, K2, qHF2_predict, cmap='viridis', edgecolor='none');
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(Wnn2, Knn2, qHFnn2, cmap='viridis', edgecolor='none');





Valid_Ratio2=20  #90% test data and 10% validation data
Wnn2v=Wnn2[ ::Valid_Ratio2]                          #takes every Ratio-th entry of W1 
Wnn2t=delete(Wnn2, slice(None, None, Valid_Ratio2))  #removes every Ratio-th entry of W1
Knn2v=Knn2[ ::Valid_Ratio2]
Knn2t=delete(Knn2, slice(None, None, Valid_Ratio2))
qHFnn2v=qHFnn2[ ::Valid_Ratio2]
qHFnn2t=delete(qHFnn2, slice(None, None, Valid_Ratio2))
x_train2=concatenate((Wnn2t[:,newaxis],Knn2t[:,newaxis]),1)
x_valid2=concatenate((Wnn2v[:,newaxis],Knn2v[:,newaxis]),1)
y_train2=qHFnn2t[:,newaxis]
y_valid2=qHFnn2v[:,newaxis]




###########################
#x_train2w=hstack((W1,W2))
#x_train2k=hstack((K1,K2))
#x_train2=concatenate((x_train2w[:,newaxis],x_train2k[:,newaxis]),1)
#y_train2=hstack((qHF1,qHF2_predict.reshape(-1)))
#y_train2=reshape(y_train2,(y_train2.size,1))

print('NN2 x_train2 shape (MT x 2):', x_train2.shape)
print('NN2 y_train2 shape:', y_train2.shape)
print('NN2 x_valid2 shape (MV x 2):', x_valid2.shape)
print('NN2 y_valid2 shape:', y_valid2.shape)


model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = x_train2.shape[1:]),    
    tf.keras.layers.Dense(30, activation =tf.nn.relu),
    tf.keras.layers.Dense(30, activation =tf.nn.relu),
    tf.keras.layers.Dense(30, activation =tf.nn.relu),
    tf.keras.layers.Dense(30, activation =tf.nn.relu),
    tf.keras.layers.Dense(1,activation=None)
    ])

learn_rate2=0.004
no_epoch=200
no_batch=50 
start_time = time.clock() 
#keras.optimizers.Adam(lr=learn_rate2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
#model2.compile(loss='mean_squared_error', optimizer='adam')
#rms_optimizer = tf.keras.optimizers.RMSprop(0.01)
#sgd_optimizer2 = tf.keras.optimizers.SGD(lr=learn_rate2, decay=1e-3, momentum=0.9, nesterov=True)
adam_optimizer2 = tf.keras.optimizers.Adam(lr = learn_rate2)
model2.compile(loss='mean_squared_error', optimizer=adam_optimizer2)


reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.98,patience=200) 
history = model2.fit(x_train2, y_train2, epochs = no_epoch, validation_data = (x_valid2,y_valid2), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr2]) 
#history = model2.fit(x_train2, y_train2, epochs = no_epoch, batch_size = no_batch, verbose = 0)  
end_time = time.clock()
CpuTimeT2 = (end_time - start_time)
print('DNN Train time:', CpuTimeT2)



history_dict = history.history
history_dict.keys()
TrainLoss = history_dict['loss']
ValLoss = history_dict['val_loss']
epoch_vec=range(1,len(TrainLoss)+1)
plt.figure(dpi=100)
plt.plot(epoch_vec,TrainLoss, label = 'Train-Loss')
plt.plot(epoch_vec,ValLoss, label = 'Val-Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

train_loss = model2.evaluate(x_train2, y_train2)     # Evaluate: Returns the loss value for the model in test mode.
print('Train loss:', train_loss)
print('Val Loss:', ValLoss[-1])
print('learning rate:', learn_rate2)
print('epochs:', no_epoch)
print('batch_size:', no_batch)





Wnn2=hstack((W1,W2))-10.0
Knn2=hstack((K1,K2))*0.5-2.0
qHFnn2=hstack((qHF1,qHF2_predict.reshape(-1)))
WKnn=concatenate((Wnn2[:,newaxis],Knn2[:,newaxis]),1)
start_time = time.clock() 
qHF_predict = model2.predict(WKnn)
end_time = time.clock()
CpuTimeP2 = (end_time - start_time)/M



adict = {}
adict['W1'] = W1
adict['K1'] = K1
adict['W2'] = W2
adict['K2'] = K2
adict['qLF1'] = qLF1
adict['qLF2'] = qLF2
adict['qHF1'] = qHF1
adict['qHF2'] = qHF2
adict['qHF2_predict'] = qHF2_predict
adict['x_train2w'] = Wnn2t
adict['x_train2k'] = Knn2t
adict['y_train2'] = qHFnn2t
adict['Wnn2'] = Wnn2
adict['Knn2'] = Knn2
adict['qHF_predict'] = qHF_predict
#sio.savemat('PDE_2a_matlab.mat', adict)


#if False: '''
###################################################


for k in range(1,1+10):
    random.seed(k+900)
    Nmc = 15*10**1
    
    Yrand = random.rand(Nmc,2)
    #Wrand=Yrand[:,0]+10
    #Krand=2*Yrand[:,1]+4
    Wrand=Yrand[:,0]
    Krand=Yrand[:,1]
    WKrand=concatenate((Wrand[:,newaxis],Krand[:,newaxis]),1)

    start_time = time.clock() 
    qHF_predict = model2.predict(WKrand)
    end_time = time.clock()
    PredictTimeDNN = (end_time - start_time)/Nmc
    
    Error_MCMFNN = abs(EQ_exact-mean(abs(qHF_predict)))
    
    print('Total error:', Error_MCMFNN)
    

print('Predict Time DNN:', PredictTimeDNN)
print('Predict Time SNN:', PredictTimeSNN)



#Mean Square Error
    

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
        seed(kk+1)
        Yrand = random.rand(Nmc,2)
        #Wrand=Yrand[:,0]
        #Krand=Yrand[:,1]
        #WKrand=concatenate((Wrand[:,newaxis],Krand[:,newaxis]),1)
        #qHF_predict_k = model2.predict(WKrand)
        Wrand=Yrand[:,0]+10
        Krand=(Yrand[:,1]+2)*2
        qHF_predict_k = model2.predict(Yrand)
        qNN=abs(qHF_predict_k)
        
        QExact=np.zeros((Nmc,))
        for i in range(0,Nmc):
            QExact[i]=u_exact(T,xQ,xQ,Wrand[i],Krand[i],Krand[i])

        qExact=abs(QExact)
        MSE=MSE+mean((qExact-qNN.reshape(-1))**2)
        
    MSE=MSE/S
    print('MSE error:', MSE)   