import numpy as np


num_samples = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

types = ["RMFNN_training_data",
        "MF_test_data",
        "MF_validation_data",
        "RMFNN_test_pred",
        "RMFNN_val_pred",
        "RMFNN_LF_pred_on_test_inputs",
        "RMFNN_LF_pred_on_validation_inputs"]

new_types = ["RLNN_training_data",
        "RLNN_test_data",
        "RLNN_validation_data",
        "RLNN_test_pred",
        "RLNN_val_pred",
        "RLNN_LF_pred_on_test_inputs",
        "RLNN_LF_pred_on_validation_inputs"]

both_types =  list(zip(types,new_types))

idx1 = [0,2]

for i in num_samples:
    for t,tnew in both_types:
        data = np.loadtxt(f'{t}_{i}.txt', dtype = float)
        
        if t == "RMFNN_training_data":
            data_new = data[:,idx1]
            np.savetxt(f'{tnew}_{i}.txt', data_new)
        elif t=="MF_test_data" or t=="MF_validation_data":
            data_new = data[:,idx1]
            np.savetxt(f'{tnew}_{i}.txt', data_new)
        else:
            data_new = data.copy()
            np.savetxt(f'{tnew}_{i}.txt', data_new)





