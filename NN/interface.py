import torch
'''
Este módulo implementa uma interface iterativa para realizar
consultas na rede treinada.
'''

def predict_results(network, device, dim, dhl, tgo, tgp):
    entry = torch.as_tensor([dim, dhl, tgo, tgp]).float().to(device)
    output = network(entry)
    return output.ge(0.5).item()

def use_network(path, device):
    network = torch.load(path)
    message1 = '>>> Entre com os resultados dos exames necessários (apenas pontos e dígitos)\n'
    message2 = '>>> [Formato: D-Dímero/Desidrogenase Láctica (DHL)/TGO/TGP]\n>>> '
    answer = input(message1+message2).split('/')
    if answer[0] == 'e':
        return False
    dim = float(answer[0])
    dhl = float(answer[1])
    tgo = float(answer[2])
    tgp = float(answer[3])
    prediction == predict_results(network, device, dim, dhl, tgo, tgp)
    print('Previsão de exames de PCR, IgG, e IgM:\n', prediction)
    return True

def use_iteratively(path, device):
    print('>>> Modelo carregado de ' + path)
    print('>>> (use "e" para sair)\n')
    keep_going = True
    while keep_going:
        keep_going = use_network(path, device)
