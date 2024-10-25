from numpy.random import rand
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from numpy import *
from numpy import sin
from numpy import cos 
from numpy import mean
from numpy import std
from numpy import linspace
from numpy import argmax
import matplotlib
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import scipy.io as sio #To save and load mat files for MATLAB

from keras.regularizers import l2
#########################################################################
## 
#########################################################################

#initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def u_exact(y, t):
    return 0.5 + 2*sin(12*y) + 6*sin(2*t)*sin(10*y)*(1+2*y**2)

def absu_exact(y,t):
    return abs(0.5 + 2*sin(12*y) + 6*sin(2*t)*sin(10*y)*(1+2*y**2))

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
seeds = [1+11*s for s in range(20)]
tols = [1e-2, 1e-3, 1e-4]
Nmcs = [140*10**3, 140*10**5, 140*10**7]
hLFs = [0.5, 0.25, 0.1]
hHFs = [0.1, 0.025, 0.01]
Ns = [241,801,3201]
ResNN_Nepochs = [100,1500,5000]
ResNN_Nbatches = [10,30,50]
DNN_Nepochs = [400, 8000, 20000]
DNN_Nbatches = [40, 80, 200]

results = {}
params = list(zip(tols, Nmcs, hLFs, hHFs, Ns, ResNN_Nepochs, ResNN_Nbatches, DNN_Nepochs, DNN_Nbatches))


for (tol, Nmc,  hLF, hHF, num_pts, ResNN_epoch_number, ResNN_batches, DNN_epoch_number, DNN_batches) in params:
    if tol != 0.01:
        continue
    if rank == 0:

        results[str(tol)] = {} 
        results[str(tol)]['hLF'] = hLF
        results[str(tol)]['hHF'] = hHF
        results[str(tol)]['N'] = num_pts
        results[str(tol)]['ResNN_Nepoch'] = ResNN_epoch_number
        results[str(tol)]['ResNN_Nbatches'] = ResNN_batches
        results[str(tol)]['DNN_Nepoch'] = DNN_epoch_number
        results[str(tol)]['DNN_Nbatches'] = DNN_batches
        results[str(tol)]['W_HF'] = {}
        results[str(tol)]['W_HF']['data'] = zeros(len(seeds),)
        results[str(tol)]['HFMC'] = {}
        results[str(tol)]['HFMC']['data'] = zeros(len(seeds),)
        results[str(tol)]['W_LF'] = {}
        results[str(tol)]['W_LF']['data'] = zeros(len(seeds),)
        results[str(tol)]['ResNNTrainTime'] = {}
        results[str(tol)]['ResNNTrainTime']['data'] = zeros(len(seeds),)
        results[str(tol)]['ResNNPredictTime'] = {}
        results[str(tol)]['ResNNPredictTime']['data'] = zeros(len(seeds),)
        results[str(tol)]['enlarged_data'] = {}
        results[str(tol)]['enlarged_data']['data'] = zeros((len(seeds), num_pts))
        results[str(tol)]['DNNTrainTime'] = {}
        results[str(tol)]['DNNTrainTime']['data'] = zeros(len(seeds),)
        results[str(tol)]['DNNPredictTime'] = {}
        results[str(tol)]['DNNPredictTime']['data'] = zeros(len(seeds),)
        results[str(tol)]['DNNPrediction'] = {}
        results[str(tol)]['DNNPrediction']['data'] = zeros((len(seeds),1001))
        results[str(tol)]['TotalTime'] = {}
        results[str(tol)]['TotalTime']['data'] = zeros(len(seeds),)
        results[str(tol)]['rel_error'] = {}
        results[str(tol)]['rel_error']['data'] = zeros(len(seeds)*4,)
        results[str(tol)]['MSE'] = {}
        results[str(tol)]['MSE']['data'] = zeros(len(seeds),)
    
    comm.Barrier()
    for trial, seed in enumerate(seeds):
        tf.keras.utils.set_random_seed(seed)
        
        if rank == 0:

            print(f'working on tol = {tol} trial {trial}')

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

            results[str(tol)]['W_HF']['data'][trial] = CpuTimeHF
            results[str(tol)]['HFMC']['data'][trial] = CpuTimeHF*Nmc
            results[str(tol)]['W_LF']['data'][trial] = CpuTimeLF

            N = int(num_pts)
            yL=linspace(-1,1,N)


            Ratio=10
            yC=yL[ ::Ratio]                            #takes every Ratio-th entry of yL 
            yLnC=delete(yL, slice(None, None, Ratio))  #removes every Ratio-th entry of yL



            qLnC=qLHF(yLnC,T,hLF)
            qLC=qLHF(yC,T,hLF)
            qHC=qLHF(yC,T,hHF)
     
            M=yC.size  #NI in paper; no. of common points
            R=N-M

            print('N: ', N)
            print('M: ', M)
            print('R: ', R)


            results[str(tol)]['NI'] = M


            '''
            plt.figure()
            plt.figure(dpi=300)
            plt.plot(Y,qLF_true,'m-',linewidth=0.3,label='$Q_{LF}$')
            plt.plot(Y,qHF_true,'b-',linewidth=0.3,label='$Q_{HF}$')
            qLF=qLHF(yL,T,hLF)
            plt.plot(yL,qLF,'mo',markersize=1, marker = 'v', fillstyle = 'none', label='available LF data')
            plt.plot(yC,qHC,'bo',markersize=2,label='available HF data')
            plt.xlabel(r'$\theta$', fontsize = 14)
            #plt.ylabel(r'$Q(\theta)$', fontsize = 14)
            plt.legend(fontsize = 10)
            #plt.savefig('ODE_train_data.png', format = 'png', dpi = 300, bbox_inches = 'tight')
            plt.savefig('ODE_train_data.pdf', format = 'pdf', bbox_inches = 'tight')
            plt.show()
            plt.close()
            '''

            ##############################################################
            '''
            ResNN to compute QHF-QLF
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
                tf.keras.layers.Dense(10, activation =my_leaky_relu),
                tf.keras.layers.Dense(10, activation =my_leaky_relu),
                tf.keras.layers.Dense(1,activation=None)
                ])

            learn_rate=0.004
            no_epoch = ResNN_epoch_number
            no_batch = ResNN_batches
            start_time = time.process_time() 
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
            model_shallow.compile(loss='mean_squared_error', optimizer=adam_optimizer)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95,patience=200) 
            history = model_shallow.fit(x_train, y_train, epochs = no_epoch, validation_data = (x_valid,y_valid), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr]) 
            end_time = time.process_time()
            TrainTimeResNN = (end_time - start_time)
            print('TrainTime-ResNN: ', TrainTimeResNN)

            results[str(tol)]['ResNNTrainTime']['data'][trial] = TrainTimeResNN

            history_dict = history.history
            history_dict.keys()
            TrainLoss = history_dict['loss']
            ValLoss = history_dict['val_loss']
            epoch_vec=range(1,len(TrainLoss)+1)
            
            '''
            plt.figure()
            plt.figure(dpi=300)
            plt.plot(epoch_vec,TrainLoss, label = 'TrainLoss')
            plt.plot(epoch_vec,ValLoss, label = 'ValLoss')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.grid(True)
            plt.show()
            '''

            print('epochs:', no_epoch)
            print('batch_size:', no_batch)

            ################################
            #Use ResNN to generate the remaining R=N-M HF data
            qLnC_scaled=qLnC/10
            x_predict=concatenate((yLnC[:,newaxis],qLnC_scaled[:,newaxis]),1)
            FnC_predict = model_shallow.predict(x_predict)
            qHnC_predict=FnC_predict.reshape(-1)+qLnC
            

            x_predict_aux = concatenate([x_predict for j in range(1000)],0)
            start_time = time.process_time() 
            FnC_predict_aux = model_shallow.predict(x_predict_aux)
            end_time = time.process_time()
            PredictTimeResNN = (end_time - start_time)/len(x_predict_aux)
            print(PredictTimeResNN)


            
            ################################
            #Stack up all HF data 
            x_DNN=hstack((yC,yLnC))
            y_DNN=hstack((qHC,qHnC_predict))

            results[str(tol)]['enlarged_data']['data'][trial,:] = y_DNN
            results[str(tol)]['enlarged_data_x'] = x_DNN
            results[str(tol)]['QHF_truth'] = qHF_true
            results[str(tol)]['QHF_train_data'] = qHC
            results[str(tol)]['QHF_points'] = yC

            '''
            plt.figure()
            plt.figure(dpi=300)
            #plt.figure(dpi=1000)
            plt.plot(Y,qHF_true,'b-',linewidth=1,label=r'$Q_{HF}$')
            plt.plot(yC,qHC,'bo',markersize=3,label='available HF data')
            plt.plot(x_DNN,y_DNN,'rx',markersize=2, marker = 'x', label='generated HF data (by ResNN)')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$Q(\theta)$')
            plt.legend()
            #plt.savefig('ODE_enlarged_train_data.png', format = 'png', dpi = 300, bbox_inches = 'tight')
            plt.savefig('ODE_enlarged_train_data.pdf', format = 'pdf', bbox_inches = 'tight')
            plt.show()
            '''

            ##############################################################
            '''
              DNN to compute QHF
            '''
            ##############################################################

            print('x_DNN shape:', x_DNN.shape)
            print('y_DNN shape:', y_DNN.shape)



            I=range(1,N-1,20)
            x_valid0=x_DNN[I]
            x_train0=delete(x_DNN,I)
            y_valid0=y_DNN[I]
            y_train0=delete(y_DNN,I)




            print('DNN x_train0 shape:', x_train0.shape)
            print('DNN y_train0 shape:', y_train0.shape)
            print('DNN x_valid0 shape:', x_valid0.shape)
            print('DNN y_valid0 shape:', y_valid0.shape)



            model_deep = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape = x_train0.shape[1:]),
                tf.keras.layers.Dense(20, activation =my_leaky_relu),
                tf.keras.layers.Dense(20, activation =my_leaky_relu),
                tf.keras.layers.Dense(20, activation =my_leaky_relu),
                tf.keras.layers.Dense(20, activation =my_leaky_relu),
                tf.keras.layers.Dense(1,activation=None)
                ])

            learn_rate=0.004 #0.004
            no_epoch = DNN_epoch_number
            no_batch = DNN_batches
            start_time = time.process_time() 
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
            model_deep.compile(loss='mean_squared_error', optimizer=adam_optimizer)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95,patience=300) 
            history = model_deep.fit(x_train0, y_train0, epochs = no_epoch, validation_data = (x_valid0,y_valid0), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr]) 
            end_time = time.process_time()
            TrainTimeDNN = (end_time - start_time)
            print('TrainTime-DNN: ', TrainTimeDNN)

            results[str(tol)]['DNNTrainTime']['data'][trial] = TrainTimeDNN

            history_dict = history.history
            history_dict.keys()
            TrainLoss = history_dict['loss']
            ValLoss = history_dict['val_loss']
            epoch_vec=range(1,len(TrainLoss)+1)
            
            '''
            plt.figure()
            plt.figure(dpi=300)
            plt.plot(epoch_vec,TrainLoss, label = 'TrainLoss')
            plt.plot(epoch_vec,ValLoss, label = 'ValLoss')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.grid(True)
            plt.show()
            '''
      
            print('learning rate:', learn_rate)
            print('epochs:', no_epoch)
            print('batch_size:', no_batch)

            ###################################

            qHF_predict = model_deep.predict(Y)
            
            results[str(tol)]['DNNPrediction']['data'][trial, :] = qHF_predict.reshape(1001,)
            results[str(tol)]['DNNPrediction_x'] = Y
            '''
            plt.figure()
            plt.figure(dpi=300)
            #plt.figure(dpi=1000)
            plt.plot(Y,qHF_true,'b-',linewidth=1,label=r'$Q_{HF}$')
            plt.plot(Y,qHF_predict,'r--',linewidth=1,label=r'$Q_{RMFNN}$')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$Q(\theta)$')
            plt.legend()
            plt.savefig('ODE_RMFNN_prediction.png', format = 'png', dpi = 300, bbox_inches = 'tight')
            plt.show()
            '''

            print(x_train0.shape[1:])
            model_deep.save_weights('DNN.weights.h5')

        comm.Barrier()
        ghost = zeros(229,)
        model_deep = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = ghost.shape[1:]),
            tf.keras.layers.Dense(20, activation =my_leaky_relu),
            tf.keras.layers.Dense(20, activation =my_leaky_relu),
            tf.keras.layers.Dense(20, activation =my_leaky_relu),
            tf.keras.layers.Dense(20, activation =my_leaky_relu),
            tf.keras.layers.Dense(1,activation=None)
            ])
        model_deep.load_weights('DNN.weights.h5')

        num_rel_errs = 4
        if tol == 0.01:
            S = 10
        else:
            S = 100
        for j in range(1,1+num_rel_errs):
            running_mean = 0
            np.random.seed(j*21)
            Nmck = Nmc // size
            if rank == size - 1:
                Nmck += Nmc % size
            for kk in range(1,1+S):
                Nmckk = round(Nmck/S)
                np.random.seed(rank*j*kk+101)
                YY = 2*rand(Nmckk,1)-1
                YY = reshape(YY,Nmckk)
                if rank == 0:
                    start_time = time.process_time()    
                    qHF_predict_kk = model_deep.predict(YY)
                    end_time = time.process_time()
                    PredictTimeDNN = (end_time - start_time)/Nmckk
                    running_mean = running_mean + mean(abs(qHF_predict_kk))
                else:
                    qHF_predict_kk = model_deep.predict(YY)
                    running_mean = running_mean + mean(abs(qHF_predict_kk))
            running_mean = running_mean / S
            gather_means = comm.reduce(running_mean, MPI.SUM, root=0)
            if rank == 0:
                full_mean = gather_means/size
                Error_MCRMFNN = abs(E_u_exact - full_mean)/abs(E_u_exact)
                results[str(tol)]['rel_error']['data'][trial*num_rel_errs + j - 1] = Error_MCRMFNN
                print('Total Error:', Error_MCRMFNN)
        comm.Barrier()
        
        MSE = 0
        if rank==0:
            S_mse = 100
            Nmc_mse = 10**6
            for kk in range(1,1+S_mse):
                Nmc_msek = round(Nmc_mse/S_mse)
                np.random.seed(kk+3)
                YY = 2*rand(Nmc_msek,1)-1
                YY = reshape(YY,Nmc_msek)
                qHF_predict_kk = model_deep.predict(YY)
                qNN=abs(qHF_predict_kk)
                qExact=absu_exact(YY,T)
                MSE=MSE+mean((qExact-qNN.reshape(-1))**2)
            MSE = MSE / S_mse

            print('MSE error:', MSE)
            results[str(tol)]['MSE']['data'][trial] = MSE
            print('Predict Time DNN:', PredictTimeDNN)
            results[str(tol)]['DNNPredictTime']['data'][trial] = PredictTimeDNN*Nmc
            print('Predict Time ResNN:', PredictTimeResNN)
            results[str(tol)]['ResNNPredictTime']['data'][trial] = PredictTimeResNN*Nmc
            Total_cost=N*CpuTimeLF+M*CpuTimeHF+TrainTimeDNN+TrainTimeResNN+Nmc*PredictTimeDNN+R*PredictTimeResNN
            print('Total cost:', Total_cost)
            results[str(tol)]['TotalTime']['data'][trial] = Total_cost
            print(results[str(tol)]['rel_error']['data'])
        
        
            results[str(tol)]['W_HF']['mean'] =  mean(results[str(tol)]['W_HF']['data']) 
            results[str(tol)]['W_HF']['std_dev'] =  std(results[str(tol)]['W_HF']['data'])
            
            results[str(tol)]['HFMC']['mean'] =  mean(results[str(tol)]['HFMC']['data']) 
            results[str(tol)]['HFMC']['std_dev'] =  std(results[str(tol)]['HFMC']['data'])

            results[str(tol)]['W_LF']['mean'] =  mean(results[str(tol)]['W_LF']['data']) 
            results[str(tol)]['W_LF']['std_dev'] =  std(results[str(tol)]['W_LF']['data'])

            results[str(tol)]['ResNNTrainTime']['mean'] = mean(results[str(tol)]['ResNNTrainTime']['data'])
            results[str(tol)]['ResNNTrainTime']['std_dev'] = std(results[str(tol)]['ResNNTrainTime']['data'])

            results[str(tol)]['ResNNPredictTime']['mean'] = mean(results[str(tol)]['ResNNPredictTime']['data'])
            results[str(tol)]['ResNNPredictTime']['std_dev'] = std(results[str(tol)]['ResNNPredictTime']['data'])

            results[str(tol)]['enlarged_data']['mean'] = mean(results[str(tol)]['enlarged_data']['data'], axis=0)
            results[str(tol)]['enlarged_data']['std_dev'] = std(results[str(tol)]['enlarged_data']['data'], axis=0)

            results[str(tol)]['DNNTrainTime']['mean'] = mean(results[str(tol)]['DNNTrainTime']['data'])
            results[str(tol)]['DNNTrainTime']['std_dev'] = std(results[str(tol)]['DNNTrainTime']['data'])

            results[str(tol)]['DNNPredictTime']['mean'] = mean(results[str(tol)]['DNNPredictTime']['data'])
            results[str(tol)]['DNNPredictTime']['std_dev'] = std(results[str(tol)]['DNNPredictTime']['data'])

            results[str(tol)]['DNNPrediction']['mean'] = mean(results[str(tol)]['DNNPrediction']['data'], axis=0)
            results[str(tol)]['DNNPrediction']['std_dev'] = std(results[str(tol)]['DNNPrediction']['data'], axis=0)

            results[str(tol)]['TotalTime']['mean'] = mean(results[str(tol)]['TotalTime']['data'])
            results[str(tol)]['TotalTime']['std_dev'] = std(results[str(tol)]['TotalTime']['data'])

            results[str(tol)]['rel_error']['mean'] = mean(results[str(tol)]['rel_error']['data'])
            results[str(tol)]['rel_error']['std_dev'] = std(results[str(tol)]['rel_error']['data'])

            results[str(tol)]['MSE']['mean'] = mean(results[str(tol)]['MSE']['data'])
            results[str(tol)]['MSE']['std_dev'] = std(results[str(tol)]['MSE']['data'])
        comm.Barrier()
    comm.Barrier()
    if rank == 0:    
        pickle.dump(results, open('../post_process_and_plots/results_RMFNN_ODE_final.pkl', 'wb'))
    comm.Barrier()



