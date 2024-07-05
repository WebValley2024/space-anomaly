from solver import Solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def train(config_train):
    
    #Training
    Solver(config_train).train()

    dataset_name = config_train['dataset_name']
    
    config_test = config_train.copy()
    config_test['mode'] = 'test'
    config_test['step'] = config_test['win_size']
    config_test['test_model'] = f'test_{config_test["ID"]}/space_checkpoint.pth'

    #Testing
    Solver(config_test).test()



def plot_res(config_train, tresh, tresh_folder):

    dataset_name = config_train['dataset_name']
    
    config_test = config_train.copy()
    config_test['mode'] = 'test'
    config_test['step'] = config_test['win_size']
    config_test['test_model'] = f'test_{config_test["ID"]}/space_checkpoint.pth'
    config_test["quantile_treshold"] = tresh


    if not os.path.exists(f'test_{config_test["ID"]}/{tresh_folder}'):
        os.makedirs(f'test_{config_test["ID"]}/{tresh_folder}')

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

    fig.savefig(f'test_{config_test["ID"]}/{tresh_folder}/histograms.png')

    trh = np.quantile(train, config_test['quantile_treshold'])

    plt.close()

    #tresholded histograms
    fig, ax = plt.subplots(2,1,figsize=(10, 6))

    ax[0].hist(train[train>trh], bins=100)
    ax[0].set_title('Train set treshold: '+str(trh))
    ax[1].hist(test[test>trh], bins=100)
    ax[1].set_title('Test set')
    fig.savefig(f'test_{config_test["ID"]}/{tresh_folder}/histograms_trsh.png')
    plt.close()

    #load dataset
    test_path = os.path.join(config_test['data_path'], dataset_name[0])
    train_path = os.path.join(config_test['data_path'], dataset_name[1])

    test_set=pd.read_csv(test_path)
    train_set=pd.read_csv(train_path)

    if 'time' not in test_set.keys():
        test_set.rename(columns={test_set.keys()[0]:'time'}, inplace=True)
    
    if 'time' not in train_set.keys():
        train_set.rename(columns={train_set.keys()[0]:'time'}, inplace=True)

    test_set.set_index('time', inplace=True)
    train_set.set_index('time', inplace=True)

    comp_set=pd.concat([test_set, train_set])
    comp_set.sort_values(by='time', inplace=True)


    comp_set.index = pd.to_datetime(comp_set.index)

    fig, ax = plt.subplots(len(comp_set.columns)+1,1,figsize=(110, 85))

    points = np.where(train>trh)[0]

    points_test = np.where(test>trh)[0]
    points_test = points_test + len(train)

    for i in range(1, config_test["input_c"]+1):
        ax[i].plot(range(len(comp_set[comp_set.keys()[i-1]].values)),comp_set[comp_set.keys()[i-1]].values)
        ax[i].set_title(comp_set.keys()[i-1])

    for l in range(0, config_test["input_c"]+1):

        for x in points:
            ax[l].axvline(x=x, color='r',alpha=0.5)

        for x in points_test:
            ax[l].axvline(x=x, color='g',alpha=0.5)


    ax[0].plot(train)
    ax[0].set_title(f'Model output, {config_test["quantile_treshold"]} treshold, anomalies_found: {len(points)+len(points_test)}')
    ax[0].plot(range(len(train),len(train)+len(test)),test)


    fig.savefig(f'test_{config_test["ID"]}/{tresh_folder}/final_time_series.png')

    plt.close("all")




config_train = {
    
    'ID': 8,                            # folder ID
    'lr': 0.0001,                       # learning rate
    'num_epochs': 1,#1500,                  # number of epochs
    'k': 3,                             # k value
    'win_size': 90,                     # window size    
    'input_c': 10,                       # input features
    'output_c': 10,                      # output features
    'batch_size': 256,                   # batch size
    'dataset': 'space',                 # dataset
    'mode': 'train',                    # mode
    'data_path': 'dataset/space/3',     # data path
    'model_save_path': 'checkpoints',   # model save path
    'step':20,                          # stride    
    'test_model': None,                 # is_testing
    'dataset_name': ['HEPP_LD_test.csv', 'HEPP_LD_train.csv'], # dataset name 0=test, 1=train
    
    # model parameters
    'e_layers': 3,                          
    'n_heads': 8,                          
    'd_ff': 512,
    'd_model': 512,                     
    'dropout': 0,

    #plot parameters
    'quantile_treshold': [["4_sigm", 0.999],["5_sigm", 0.99987]]

}

# HEPP_L media conteggi oky

# HEPP_L media conteggi + media altri 6 oky

# HEPD cont prot el oky

# tutti e due oky

# modello scompattato heppl oky

# modello scompattato heppl + hepd oky

# [['HEPP_L_test.csv', 'HEPP_L_train.csv'], "dataset/space/1", 2]
# [['HEPP_L_test.csv', 'HEPP_L_train.csv'], "dataset/space/1", 8]
# [['HEPP_D_data_test.csv', 'HEPP_D_data_train.csv'], "dataset/space/2", 2]
# [['HEPP_LD_test.csv', 'HEPP_LD_train.csv'], "dataset/space/3", 10]
# [['HEPP_L_split_test.csv', 'HEPP_L_split_train.csv'],"dataset/space/4", 64]
# [['HEPP_LD_split_test.csv', 'HEPP_LD_split_train.csv'],"dataset/space/5", 66]

# [['EFD_var_nomean_1esp_test.csv', 'EFD_var_nomean_1esp_train.csv'], "dataset/space/6", 3]
# [['EFD_var_nomean_full_test.csv', 'EFD_var_nomean_full_train.csv'], "dataset/space/7", 9]
# [['EFD_HEPPL_test.csv', 'EFD_HEPPL_train.csv'], "dataset/space/8", 11]
# [['EFD_HEPPL_HEPD_test.csv', 'EFD_HEPPL_HEPD_train.csv'], "dataset/space/9", 13]

dataset_names = [[['EFD_var_nomean_1esp_test.csv', 'EFD_var_nomean_1esp_train.csv'], "dataset/space/6", 3], [['EFD_var_nomean_full_test.csv', 'EFD_var_nomean_full_train.csv'], "dataset/space/7", 9], [['EFD_HEPPL_test.csv', 'EFD_HEPPL_train.csv'], "dataset/space/8", 11], [['EFD_HEPPL_HEPD_test.csv', 'EFD_HEPPL_HEPD_train.csv'], "dataset/space/9", 13]]

wind_size = [90]
dff = [512]
dpt_model = [256]
heads = [10]
e_layers = [3]
k_val=[0.1]

index = 500

for i in dataset_names:
    for w in wind_size:
        for d in dff:
            for h in heads:
                for e in e_layers:
                    for k in k_val:
                        for dpt in dpt_model:
                            
                            config_train['d_model'] = dpt

                            print(f"dataset: {i[0]}, window size: {w}, d_ff: {d}, heads: {h}, e_layers: {e}, k: {k}")

                            config_train['win_size'] = w
                            config_train['d_ff'] = d
                            config_train['n_heads'] = h
                            config_train['e_layers'] = e
                            config_train['k'] = k
                            config_train["input_c"] = i[2]
                            config_train["output_c"] = i[2]
                            config_train['dataset_name'] = i[0]
                            config_train['data_path'] = i[1]
                            config_train['ID'] = index

                            train(config_train)

                            for indx, ints in config_train['quantile_treshold']:

                                plot_res(config_train, ints, f'tresh_{indx}')

                            index+=1
