from solver import Solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

config_train = {
    
    'ID': 1,                            # folder ID
    'lr': 0.0001,                       # learning rate
    'num_epochs': 100,                 # number of epochs
    'k': 3,                             # k value
    'win_size': 50,                     # window size    
    'input_c': 8,                       # input features
    'output_c': 8,                      # output features
    'batch_size': 16,                   # batch size
    'dataset': 'space',                 # dataset
    'mode': 'train',                    # mode
    'data_path': 'dataset/space',       # data path
    'model_save_path': 'checkpoints',   # model save path
    'step':50,                          # stride    
    'test_model': None,                 # is_testing
    
    # model parameters
    'e_layers': 3,                          
    'n_heads': 8,                          
    'd_ff': 512,
    'd_model': 512,                     
    'dropout': 0,

    #plot parameters
    'quantile_treshold': 0.999937

}

dataset_name = ['HEPP_L_data_test.csv', 'HEPP_L_data_train.csv']




config_test = config_train.copy()
config_test['mode'] = 'test'
config_test['step'] = config_test['win_size']
config_test['test_model'] = f'test_{config_test["ID"]}/space_checkpoint.pth'

#Training
Solver(config_train).train()

#Testing
Solver(config_test).test()


#Plotting


test = np.loadtxt(f'test_{config_test["ID"]}/train_energy_test.csv', delimiter=',')
train = np.loadtxt(f'test_{config_test["ID"]}/train_energy.csv', delimiter=',')

#HISTOGRAMS
fig, ax = plt.subplots(2,1,figsize=(10, 6))

ax[0].hist(test, bins=100)
ax[0].set_title('Test set')
ax[0].set_yscale('log')
ax[1].hist(train, bins=100)
ax[1].set_title('Train set')
ax[1].set_yscale('log')
fig.savefig(f'test_{config_test["ID"]}/histograms.png')

trh = np.quantile(train, config_test['quantile_treshold'])

plt.close()

#tresholded histograms
fig, ax = plt.subplots(2,1,figsize=(10, 6))

ax[0].hist(train[train>trh], bins=100)
ax[0].set_title('Train set treshold: '+str(trh))
ax[1].hist(test[test>trh], bins=100)
ax[1].set_title('Test set')
fig.savefig(f'test_{config_test["ID"]}/histograms_trsh.png')
plt.close()

#load dataset
test_path = os.path.join(config_test['data_path'], dataset_name[0])
train_path = os.path.join(config_test['data_path'], dataset_name[1])

test_set=pd.read_csv(test_path)
train_set=pd.read_csv(train_path)
test_set.set_index('time', inplace=True)
train_set.set_index('time', inplace=True)

comp_set=pd.concat([test_set, train_set])
comp_set.sort_values(by='time', inplace=True)


comp_set.index = pd.to_datetime(comp_set.index)

fig, ax = plt.subplots(len(comp_set.columns)+1,1,figsize=(100, 50))

points = np.where(train>trh)[0]

points_test = np.where(test>trh)[0]
points_test = points_test + len(train)

for i in range(1, len(comp_set.columns)+1):
    ax[i].plot(range(len(comp_set[comp_set.keys()[i-1]].values)),comp_set[comp_set.keys()[i-1]].values)
    ax[i].set_title(comp_set.keys()[i-1])

for l in range(0, len(comp_set.columns)+1):

    for x in points:
        ax[l].axvline(x=x, color='r',alpha=0.1)

    for x in points_test:
        ax[l].axvline(x=x, color='g',alpha=0.1)


ax[0].plot(train)
ax[0].set_title('Model output')

ax[0].plot(range(len(train),len(train)+len(test)),test)


fig.savefig(f'test_{config_test["ID"]}/final_time_series.png')