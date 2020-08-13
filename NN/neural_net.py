import torch
from sklearn.model_selection import train_test_split

'''
Definiremos aqui o modelo neural e as funções que usaremos para
treiná-lo. Os dados usados serão os fornecidos por pre_process.py
'''


# Classe da rede que usarei: feed-forward simples com duas camadas ocultas
class FFN_2HLayers(nn.Module):
    super(FFN_2HLayers, self).__init__()

    self.hidden1 = nn.Linear(n_features, n_hl1)
    self.hidden2 = nn.Linear(n_hl1, n_hl2)
    self.output_layer = nn.Linear(n_hl2, 3)


    def forward(self, x):
        h1 = nn.functional.relu(self.hidden1(x))
        h2 = nn.functional.relu(self.(h1))
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







