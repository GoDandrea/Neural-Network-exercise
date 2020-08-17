from PRE.pre_process import get_einstein, get_fleury, join_data
from NN.neural_net import FFN_2HLayers, split_data, train_network
from NN.interface import use_iteratively
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

    message1 = '>>> Entre "U" para utilizar a rede em '
    message2 = '\n>>> Entre "T" para treinar uma rede nova:\n>>> '
    mode = input(message1+MODEL_PATH+message2)

    if mode not in ['U', 'u', 'T', 't']:
        print('>>> Modo', mode, 'inválido. Encerrando programa.\n')
    elif (mode == 'U') or (mode == 'u'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        use_iteratively(MODEL_PATH, device)
        print('>>> Encerrando programa.\n')

    else:
        try:
            print(">>> Carregando dados prá-processados")
            processed_data = pd.read_csv('PRE/treated.csv', sep='|')
        except:
            print(">>> Dados pré-processados não encontrados")
            print(">>> Carregando dados do Einstein")
            ein_data = get_einstein('PRE/dados/einstein_exames.csv')
            print(">>> Carregando dados do Fleury")
            fleury_data = get_fleury('PRE/dados/fleury_exames.csv')
            print(">>> Unificando dados")
            processed_data = join_data(ein_data, fleury_data)
            
        print(">>> Separando dados para treino")
        splitted_data = split_data(processed_data)

        print(">>> Montando modelo")
        x_train = splitted_data[0]
        network = FFN_2HLayers(x_train.shape[1], N_HL1, N_HL2)
        optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        criteria = torch.nn.BCELoss()

        print('>>> Treinando...')
        train_network(splitted_data, network, optimizer, criteria, MODEL_PATH)
        print('>>> Modelo treinado com sucesso.\n>>> Está salvo em ' + MODEL_PATH)

