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
import scipy.io as sio
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
    ny=nx                                                  #equal number of grid points in x and y directions
    side_length = 2                                        #Length of one side of computational domain
    hx, hy = side_length/(nx-1), side_length/(ny-1)        #grid size
    x = np.linspace(-1,1,nx)                               #Spatial grid points
    y = np.linspace(-1,1,nx) 
    X, Y = np.meshgrid(x, y)
    
    #Set up the time grid to calcuate the equation.
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
        #obtain right hand side
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
    Q=u2[mQ-1,mQ-1]
    return Q

#######################################################
#######################################################
#######################################################
T=30
xQ=0.5
EQ_exact=0.358525224
seeds = [1+11*s for s in range(20)]
tols = [1e-1, 1e-2, 1e-3]
Nmcs = [150, 150*10**2, 150*10**4]
hHFs = [1/32, 1/128, 1/320]
hLFs = [1/20, 1/32, 1/40]
Ns = [3498, 4961, 8003]
NIs = [324, 451, 714]
ResNN_Nepochs = [100, 200, 1000]
ResNN_Nbatches = [50,50,50]
DNN_Nepochs = [500, 1000, 5000]
DNN_Nbatches = [50, 100, 200]
nLFs = [41, 65, 81]
nHFs = [65, 257, 641]
Mws = [106, 121, 151]
Mks = [33, 41, 53]

results = {}
params = list(zip(tols, hLFs, hHFs, Nmcs, nLFs, nHFs, Mws, Mks, Ns, NIs, ResNN_Nepochs, ResNN_Nbatches, DNN_Nepochs, DNN_Nbatches))

for (tol, hLF, hHF, Nmc, nLF, nHF, Mw, Mk, num_pts, num_hf_pts, ResNN_epoch_number, ResNN_batches, DNN_epoch_number, DNN_batches) in params:
    print(tol)
    if rank ==0:
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
            '''                
            if trial == 0:
                qHF1_aux = np.zeros((10,))
                start_time = time.process_time() 
                for i in range(0,10):
                    if i % 1 == 0:
                        print(i)
                    qHF1_aux[i]=wave_solver(T,xQ,xQ,nHF,W1[i],K1[i],K1[i])

                end_time = time.process_time()
                CpuTimeHF = (end_time - start_time)/10

                qHF1=np.zeros((M1,))
                for i in range(0,M1):
                    if i % 500 == 0:
                        print(i)
                    qHF1[i]=u_exact(T,xQ,xQ,W1[i],K1[i],K1[i])

                qLF1=np.zeros((M1,))
                for i in range(0,M1): 
                    if i % 500 == 0:
                        print(i)
                    qLF1[i]=wave_solver(T,xQ,xQ,nLF,W1[i],K1[i],K1[i])

                qLF2=np.zeros((M2,))   
                start_time = time.process_time()  
                for i in range(0,M2):
                    if i % 500 == 0:
                        print(i)
                    qLF2[i]=wave_solver(T,xQ,xQ,nLF,W2[i],K2[i],K2[i])

                end_time = time.process_time()
                CpuTimeLF = (end_time - start_time)/M2
            '''
            CpuTimeLF =  0.07
            CpuTimeHF = 0.29
            print('CPuTimeLF:', CpuTimeLF)
            print('CPuTimeHF:', CpuTimeHF)
               

            qHF1 = np.loadtxt(f'qHF1_{tol}.txt', dtype=float)
            qLF1 = np.loadtxt(f'qLF1_{tol}.txt', dtype=float)
            qLF2 = np.loadtxt(f'qLF2_{tol}.txt', dtype=float)
            
            results[str(tol)]['W_HF']['data'][trial] = CpuTimeHF
            results[str(tol)]['HFMC']['data'][trial] = CpuTimeHF*Nmc
            results[str(tol)]['W_LF']['data'][trial] = CpuTimeLF


            ###############################################################
            #Step 3

            ##############################################################
            '''
              ResNN to compute QHF-QLF = F(y, qL(y)) where y=(w,k)
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
                tf.keras.layers.Dense(1,activation=None)
                ])

            learn_rate=0.005
            no_epoch = ResNN_epoch_number
            no_batch = ResNN_batches
            start_time = time.process_time()
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
            model_shallow.compile(loss='mean_squared_error', optimizer=adam_optimizer)

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.98,patience=200) 
            history = model_shallow.fit(x_train, y_train, epochs = no_epoch, validation_data = (x_valid,y_valid), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr]) 
            end_time = time.process_time()
            TrainTimeResNN = (end_time - start_time)
            print('ResNN Train time:', TrainTimeResNN)

            results[str(tol)]['ResNNTrainTime']['data'][trial] = TrainTimeResNN

            history_dict = history.history
            history_dict.keys()
            TrainLoss = history_dict['loss']
            ValLoss = history_dict['val_loss']
            epoch_vec=range(1,len(TrainLoss)+1)

            '''
            plt.figure(dpi=100)
            plt.plot(epoch_vec,TrainLoss, label = 'Train-Loss')
            plt.plot(epoch_vec,ValLoss, label = 'Val-Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.grid(True)
            plt.show()
            '''


            ################################
            #Use ResNN to generate the remaining M2=M-M1 HF data

            x_predict=concatenate((W2[:,newaxis],K2[:,newaxis],qLF2[:,newaxis]),1)
            FnC_predict = model_shallow.predict(x_predict)
            qHF2_predict=FnC_predict.reshape(-1)+qLF2

            x_predict_aux = concatenate([x_predict for j in range(100)], 0)
            start_time = time.process_time() 
            FnC_predict_aux = model_shallow.predict(x_predict_aux)
            end_time = time.process_time()
            PredictTimeResNN = (end_time - start_time)/len(x_predict_aux)
            print(PredictTimeResNN)

            
            qHF2=np.zeros((M2,1)) 
            
            for i in range(0,M2):
            #    qHF2[i]=wave_solver(T,xQ,xQ,nHF,W2[i],K2[i],K2[i])
                qHF2[i]=u_exact(T,xQ,xQ,W2[i],K2[i],K2[i])

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
            no_epoch = DNN_epoch_number
            no_batch = DNN_batches
            start_time = time.process_time() 
            adam_optimizer2 = tf.keras.optimizers.Adam(learning_rate = learn_rate2)
            model2.compile(loss='mean_squared_error', optimizer=adam_optimizer2)


            reduce_lr2 = ReduceLROnPlateau(monitor='val_loss', factor=0.98,patience=200) 
            history = model2.fit(x_train2, y_train2, epochs = no_epoch, validation_data = (x_valid2,y_valid2), batch_size = no_batch, verbose = 0, callbacks=[reduce_lr2]) 
            end_time = time.process_time()
            TrainTimeDNN = (end_time - start_time)
            print('DNN Train time:', TrainTimeDNN)

            results[str(tol)]['DNNTrainTime']['data'][trial] = TrainTimeDNN



            history_dict = history.history
            history_dict.keys()
            TrainLoss = history_dict['loss']
            ValLoss = history_dict['val_loss']
            epoch_vec=range(1,len(TrainLoss)+1)

            '''
            plt.figure(dpi=100)
            plt.plot(epoch_vec,TrainLoss, label = 'Train-Loss')
            plt.plot(epoch_vec,ValLoss, label = 'Val-Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.grid(True)
            plt.show()
            ''' 
            model2.save_weights('DNN.weights.h5')
    ###################################################

        comm.Barrier()
        ghost = np.zeros((3323,2))
        model2 = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = ghost.shape[1:]),    
            tf.keras.layers.Dense(30, activation =tf.nn.relu),
            tf.keras.layers.Dense(30, activation =tf.nn.relu),
            tf.keras.layers.Dense(30, activation =tf.nn.relu),
            tf.keras.layers.Dense(30, activation =tf.nn.relu),
            tf.keras.layers.Dense(1,activation=None)
            ])
        model2.load_weights('DNN.weights.h5')


        num_rel_errs = 4
        if tol == 0.1 or tol == 0.01:
            S = 10
        else:
            S = 100

        for j in range(1, 1+num_rel_errs):
            running_mean = 0
            np.random.seed(j*21)
            Nmck = Nmc // size
            if rank == size - 1:
                Nmck += Nmc % size
            for kk in range(1, 1+S):
                Nmckk = round(Nmck/S)
                np.random.seed(rank*j*kk+101)
                Yrand = rand(Nmckk,2)

                if rank == 0:
                    start_time = time.process_time()
                    qHF_predict_kk = model2.predict(Yrand)
                    end_time = time.process_time()
                    PredictTimeDNN = (end_time - start_time)/Nmckk
                    running_mean = running_mean + mean(abs(qHF_predict_kk))
                else:

                    qHF_predict_kk = model2.predict(Yrand)
                    running_mean = running_mean + mean(abs(qHF_predict_kk))
            running_mean = running_mean / S
            gather_means = comm.reduce(running_mean, MPI.SUM, root = 0)

            if rank == 0:
                full_mean = gather_means/size
                Error_MCRMFNN = abs(EQ_exact - full_mean)
                results[str(tol)]['rel_error']['data'][trial*num_rel_errs + j - 1] = Error_MCRMFNN
                print('Total Error', Error_MCRMFNN)
        comm.Barrier()

        MSE = 0
        if rank == 0:
            S_mse = 100
            Nmc_mse = 10**6
            for kk in range(1, 1+S_mse):
                Nmc_msek = round(Nmc_mse / S_mse)
                np.random.seed(kk+3)
                Yrand = rand(Nmc_msek,2)
                Wrand=Yrand[:,0]+10
                Krand=(Yrand[:,1]+2)*2
                qHF_predict_kk = model2.predict(Yrand)
                qNN=abs(qHF_predict_kk)
                QExact=np.zeros((Nmc_msek,))
                for i in range(0,Nmc_msek):
                    QExact[i]=u_exact(T,xQ,xQ,Wrand[i],Krand[i],Krand[i])
                qExact=abs(QExact)
                MSE=MSE+mean((qExact-qNN.reshape(-1))**2)
            MSE=MSE/S_mse
            
            print('MSE error:', MSE)
            results[str(tol)]['MSE']['data'][trial] = MSE
            print('Predict Time DNN:', PredictTimeDNN)
            results[str(tol)]['DNNPredictTime']['data'][trial] = PredictTimeDNN*Nmc
            print('Predict Time ResNN', PredictTimeResNN)
            results[str(tol)]['ResNNPredictTime']['data'][trial] = PredictTimeResNN
            Total_cost = num_pts*CpuTimeLF+num_hf_pts*CpuTimeHF+TrainTimeDNN+TrainTimeResNN+Nmc*PredictTimeDNN+(num_pts-num_hf_pts)*PredictTimeResNN

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

            results[str(tol)]['DNNTrainTime']['mean'] = mean(results[str(tol)]['DNNTrainTime']['data'])
            results[str(tol)]['DNNTrainTime']['std_dev'] = std(results[str(tol)]['DNNTrainTime']['data'])

            results[str(tol)]['DNNPredictTime']['mean'] = mean(results[str(tol)]['DNNPredictTime']['data'])
            results[str(tol)]['DNNPredictTime']['std_dev'] = std(results[str(tol)]['DNNPredictTime']['data'])

            results[str(tol)]['TotalTime']['mean'] = mean(results[str(tol)]['TotalTime']['data'])
            results[str(tol)]['TotalTime']['std_dev'] = std(results[str(tol)]['TotalTime']['data'])

            results[str(tol)]['rel_error']['mean'] = mean(results[str(tol)]['rel_error']['data'])
            results[str(tol)]['rel_error']['std_dev'] = std(results[str(tol)]['rel_error']['data'])

            results[str(tol)]['MSE']['mean'] = mean(results[str(tol)]['MSE']['data'])
            results[str(tol)]['MSE']['std_dev'] = std(results[str(tol)]['MSE']['data'])
        comm.Barrier()
    comm.Barrier()

    if rank == 0:
         pickle.dump(results, open('../post_process_and_plots/results_RMFNN_PDE_final.pkl', 'wb'))
    comm.Barrier()


        
        
            

