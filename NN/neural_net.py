import torch
from torch import nn
from torch import sigmoid
from sklearn.model_selection import train_test_split

'''
Definiremos aqui o modelo neural e as funções que usaremos para
treiná-lo. Os dados usados serão os fornecidos por pre_process.py
'''

RANDOM_SEED = 42 #heh
torch.manual_seed(RANDOM_SEED)

MODEL_PATH='FFN_2HL_COVID.pt'
EPOCHS = 1000


# Classe da rede neural:
# feed-forward simples com duas camadas ocultas
class FFN_2HLayers(torch.nn.Module):
    
    def __init__(self, n_features, n_hl1, n_hl2):
        super(FFN_2HLayers, self).__init__()

        self.hidden1 = torch.nn.Linear(n_features, n_hl1)
        self.hidden2 = torch.nn.Linear(n_hl1, n_hl2)
        self.output_layer = torch.nn.Linear(n_hl2, 3)


    def forward(self, x):
        h1 = nn.functional.relu(self.hidden1(x))
        h2 = nn.functional.relu(self.hidden2(h1))
        y = sigmoid(self.output_layer(h2))
        return y


# Adequa dados para uso pela rede
def split_data(data, val_size=0.1, seed=RANDOM_SEED):

    x = data[['DIM', 'DHL', 'TGO', 'TGP']]
    y = data[['PCR', 'IGG', 'IGM']]

    # Divisão de conj. de validação e treino
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=seed)

    # Conversão para tensores do PyTorch
    x_train = torch.from_numpy(x_train.to_numpy()).float()
    y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
    x_val = torch.from_numpy(x_val.to_numpy()).float()
    y_val = torch.squeeze(torch.from_numpy(y_val.to_numpy()).float())

    return (x_train, y_train, x_val, y_val)


# Cálculo de corretude de uma previsão
def accuracy(guess, actual):
    prediction = guess.ge(.5).view(-1)
    return (actual == prediction).sum().float() / len(actual)

# Arredonda componentes de um tensor
def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)

# Auxiliar para uso de CUDA se disponível
def transfer_to_device(data, network, criteria):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    (x_train, y_train, x_val, y_val) = data

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    network = network.to(device)
    criteria = criteria.to(device)

    transfered_data = (x_train, y_train, x_val, y_val)
    return (transfered_data, network, criteria)


###############################################################################

# Função principal de treino
# Realiza o treino de uma rede usando dados, função, e critério de otimização
# fornecidos. Salva o modelo em MODEL_PATH
def train_network(data, network, optimization, criteria, path=MODEL_PATH):

    (data, network, criteria) = transfer_to_device(data, network, criteria)
    (x_train, y_train, x_val, y_val) = data

    # epoch = ciclo de treino
    for epoch in range(EPOCHS):
        
        y_guess = network(x_train)
        y_guess = torch.squeeze(y_guess)
        train_loss = criteria(y_guess, y_train) # cálculo de custo

        # resultados parciais
        if epoch % 100 == 0:
            train_acc = accuracy(y_train, y_guess)
            y_val_pred = network(x_val)
            y_val_pred = torch.squeeze(y_val_pred)
            val_loss = criteria(y_val_pred, y_val)
            val_acc = accuracy(y_val, y_val_pred)

            tr_loss = round_tensor(train_loss)
            tr_acc  = round_tensor(train_acc)
            vl_loss = round_tensor(val_loss)
            vl_acc  = round_tensor(val_acc)

            print('\nEpoch {0}'.format(epoch))
            print('Train Set --------- loss:{0}; acc:{1}'.format(tr_loss, tr_acc))
            print('Validation Set  --- loss:{0}; acc:{1}'.format(vl_loss, vl_acc))

        optimization.zero_grad()
        train_loss.backward()
        optimization.step()

    torch.save(network, path)







