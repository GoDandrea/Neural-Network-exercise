from PRE.pre_process import get_einstein, get_fleury, join_data
from NN.neural_net import FFN_2HLayers, split_data, train_network
import torch

import pandas as pd

'''
Script principal, que chama os módulos em PRE e NN para carregar, pre-processar,
e adequar os dados, depois instanciar uma rede neural Feed-Forward e treina-la.
O modelo será salvo em MODEL_PATH
'''

N_HL1 = 5
N_HL2 = 4
LEARNING_RATE = 0.001
MODEL_PATH = 'FFN_2HL_COVID.pt'

if __name__ == '__main__':

    try:
        print("Loading pre-processed data")
        processed_data = pd.read_csv('PRE/treated.csv', sep='|')
    except:
        print("pre-processed data not found.")
        print("Loading Einstein's data")
        ein_data = get_einstein('PRE/dados/einstein_exames.csv')
        print("Loading Fleury's data")
        fleury_data = get_fleury('PRE/dados/fleury_exames.csv')
        print("Joining data")
        processed_data = join_data(ein_data, fleury_data)
        
    print("Splitting data")
    splitted_data = split_data(processed_data)

    print("Building model")
    x_train = splitted_data[0]
    network = FFN_2HLayers(x_train.shape[1], N_HL1, N_HL2)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criteria = torch.nn.BCELoss()

    print('Training...')
    train_network(splitted_data, network, optimizer, criteria, MODEL_PATH)
    print('Model trained successfully.\nIt was saved at ' + MODEL_PATH)

