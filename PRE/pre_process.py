import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

'''
Esses dados exigem um trabalho amplo de pré-processamento.
Primeiro, faço uma cópia de cada fonte (Einstein e Fleury)
Filtro os pacientes (linhas) que possuem (PCR && IgG && IgM).
Depois, retiro os que não têm (D-dímero || DHL || TGO || TGP), 
as features que escolhi.
Depois, uno as três num único csv 'processado.csv'.
'''

def get_einstein(path='dados/einstein_exames.csv'):

    raw_data = pd.read_csv(path, sep="|")
    cols = ['id_paciente', 'dt_coleta', 'de_exame', 'de_analito', 'de_resultado']
    data = raw_data.loc[:,cols]
    data.rename(columns={
            'id_paciente': 'paciente',
            'dt_coleta': 'coleta',
            'de_exame': 'exame',
            'de_analito': 'analito',
            'de_resultado': 'result'
        },
        inplace=True
    )

    # Glossário
    pcr = 'PCR em tempo real para detecção de Coron'
    igg = 'COVID IgG Interp'
    igm = 'COVID IgM Interp'
    teste_ig = 'Sorologia SARS-CoV-2/COVID19 IgG/IgM'
    dim = 'Dosagem de D-Dímero'
    dhl = 'Dosagem de Desidrogenase Láctica'
    tgo = 'Dosagem de TGO'
    tgp = 'Dosagem de TGP'

    # Mantenho pacientes com PCR && IGG && IGM
    keep1 = data[data.exame == pcr].paciente
    keep2 = data[data.analito == igg].paciente
    keep3 = data[data.analito == igm].paciente
    data = data[data.paciente.isin(keep1) &
            data.paciente.isin(keep2) &
            data.paciente.isin(keep3)]

    # Retira pacientes sem exame igual a:
    keep1 = data[data.exame == dim].paciente
    keep2 = data[data.exame == dhl].paciente
    keep3 = data[data.exame == tgo].paciente
    keep4 = data[data.exame == tgp].paciente
    data = data[data.paciente.isin(keep1) &
            data.paciente.isin(keep2) &
            data.paciente.isin(keep3) &
            data.paciente.isin(keep4)]


    # Retira rows com informações irrelevantes
    data = data[(data.exame == pcr) |
                (data.analito == igg) |
                (data.analito == igm) |
                (data.exame == dim) |
                (data.exame == dhl) |
                (data.exame == tgo) |
                (data.exame == tgp)]

    df = data.copy(deep=True)
    # df.drop('analito',axis=1,inplace=True)

    # mergesort é a única opção estável de ordenação
    # em teoria é possível usar a tupla (exame, coleta, paciente) de index
    # mas não parece preservar as ordems ao mudar o index de volta
    # testes foram inconclusivos
    df['coleta'] = pd.to_datetime(df.coleta)
    df.sort_values(by='exame',inplace=True,kind='mergesort')
    df.sort_values(by='coleta',inplace=True,kind='mergesort')
    df.sort_values(by='paciente',inplace=True,kind='mergesort') 
    df.reset_index(drop=True, inplace=True)

    '''
    Criarei um novo dataframe com
    cols = ['PACIENTE', 'COLETA', 'DIM', 'DHL', 'TGO', 'TGP', 'PCR', 'IGG', 'IGM']
    usando uma lista 'lista' construída a partir dos dados do dataframe 'df'.
    '''
    lista = []
    frames = []
    cols = ['PACIENTE', 'COLETA', 'DIM', 'DHL', 'TGO', 'TGP', 'PCR', 'IGG', 'IGM']
    switcher = {dim: 2, dhl: 3, tgo: 4, tgp: 5, pcr: 6, igg: 7, igm: 8}
    blank = [None, None, None, None, None, None, None, None, None]
    builder = blank.copy()
    builder[0], builder[1] = df.paciente[0], df.coleta[1]

    for i in df.index:
        # paciente diferente
        if (df.paciente[i] != builder[0]):
            lista.append(tuple(builder.copy()))
            builder = blank.copy()
            builder[0], builder[1] = df.paciente[i], df.coleta[i]
            new_df = pd.DataFrame.from_records(lista, columns=cols)
            new_df.fillna(method='bfill', inplace=True)
            new_df.fillna(method='ffill', inplace=True)
            frames.append(new_df)
            lista = []
            builder = blank.copy()
            builder[0], builder[1] = df.paciente[i], df.coleta[i]
        # data diferente
        elif (df.coleta[i] != builder[1]):
            lista.append(tuple(builder.copy()))
            builder = blank.copy()
            builder[0], builder[1] = df.paciente[i], df.coleta[i]
        
        if (switcher.get(df.exame[i]) != None or df.exame[i] == teste_ig):
            if df.exame[i] == teste_ig:
                builder[switcher.get(df.analito[i])] = df.result[i]
            else:
                builder[switcher.get(df.exame[i])] = df.result[i]

    lista.append(tuple(builder.copy()))
    new_df = pd.DataFrame.from_records(lista, columns=cols)
    new_df.fillna(method='bfill', inplace=True)
    new_df.fillna(method='ffill', inplace=True)
    frames.append(new_df)

    processado_aux = frames[0].copy()
    for i in range(1, len(frames)):
        processado_aux = processado_aux.append(frames[i], ignore_index=True)
    processado = processado_aux.copy(deep=True)

    return processado



def get_fleury(path='dados/fleury_exames.csv'):

    raw_data = pd.read_csv(path, sep="|")
    cols = ['ID_PACIENTE', 'DT_COLETA', 'DE_EXAME', 'DE_ANALITO', 'DE_RESULTADO']
    data = raw_data.loc[:,cols]
    data.rename(columns={
            'ID_PACIENTE': 'paciente',
            'DT_COLETA': 'coleta',
            'DE_EXAME': 'exame',
            'DE_ANALITO': 'analito',
            'DE_RESULTADO': 'result'
        },
        inplace=True
    )

    # Glossário
    teste_pcr = 'NOVO CORONAVÍRUS 2019 (SARS-CoV-2), DETECÇÃO POR PCR'
    pcr = 'Covid 19, Detecção por PCR'
    igg = 'Covid 19, Anticorpos IgG, Quimioluminescência' 
    elisa = 'Covid 19, Anticorpos IgG, Elisa'
    teste_igg = 'COVID19, ANTICORPOS IgG, soro'
    igm = 'Covid 19, Anticorpos IgM, Quimioluminescência'
    teste_igm = 'COVID19, ANTICORPOS IgM, soro'
    dim = 'DIMEROS D, plasma'
    dhl = 'DESIDROGENASE LACTICA (DHL), soro'
    tgo = 'TRANSAMINASE GLUTAMICO-OXALACETICA (TGO)'
    tgp = 'TRANSAMINASE GLUTAMICO-PIRUVICA (TGP)'

    # Mantenho pacientes com PCR && IGG && IGM
    keep1 = data[data.analito == pcr].paciente
    keep2 = data[(data.analito == igg) | (data.analito == elisa)].paciente
    keep3 = data[data.analito == igm].paciente
    data = data[data.paciente.isin(keep1) &
            data.paciente.isin(keep2) &
            data.paciente.isin(keep3)]

    # Retira pacientes sem exame de dim, dhl, tgo, e tgp
    keep1 = data[data.exame == dim].paciente
    keep2 = data[data.exame == dhl].paciente
    keep3 = data[data.exame == tgo].paciente
    keep4 = data[data.exame == tgp].paciente
    data = data[data.paciente.isin(keep1) &
            data.paciente.isin(keep2) &
            data.paciente.isin(keep3) &
            data.paciente.isin(keep4)]

    # Retira rows com informações irrelevantes
    data = data[(data.analito == pcr) |
                (data.analito == igg) |
                (data.analito == elisa) |
                (data.analito == igm) |
                (data.exame == dim) |
                (data.exame == dhl) |
                (data.exame == tgo) |
                (data.exame == tgp)]

    df = data.copy(deep=True)
    df.drop('analito',axis=1,inplace=True)

    # mergesort é a única opção estável de ordenação
    # em teoria é possível usar a tupla (exame, coleta, paciente) de index
    # mas não parece preservar as ordems ao mudar o index de volta
    # testes foram inconclusivos
    df['coleta'] = pd.to_datetime(df.coleta)
    df.sort_values(by='exame',inplace=True,kind='mergesort')
    df.sort_values(by='coleta',inplace=True,kind='mergesort')
    df.sort_values(by='paciente',inplace=True,kind='mergesort') 
    df.reset_index(drop=True, inplace=True)

    '''
    Criarei um novo dataframe com
    cols = ['PACIENTE', 'COLETA', 'DIM', 'DHL', 'TGO', 'TGP', 'PCR', 'IGG', 'IGM']
    usando uma lista 'lista' construída a partir dos dados do dataframe 'df'.
    '''
    lista = []
    frames = []
    cols = ['PACIENTE', 'COLETA', 'DIM', 'DHL', 'TGO', 'TGP', 'PCR', 'IGG', 'IGM']
    switcher = {dim: 2, dhl: 3, tgo: 4, tgp: 5, teste_pcr: 6, teste_igg: 7, teste_igm: 8}
    blank = [None, None, None, None, None, None, None, None, None]
    builder = blank.copy()
    builder[0], builder[1] = df.paciente[0], df.coleta[1]

    for i in df.index:
        # paciente diferente
        if (df.paciente[i] != builder[0]):
            lista.append(tuple(builder.copy()))
            builder = blank.copy()
            builder[0], builder[1] = df.paciente[i], df.coleta[i]
            new_df = pd.DataFrame.from_records(lista, columns=cols)
            new_df.fillna(method='bfill', inplace=True)
            new_df.fillna(method='ffill', inplace=True)
            frames.append(new_df)
            lista = []
            builder = blank.copy()
            builder[0], builder[1] = df.paciente[i], df.coleta[i]
        # data diferente
        elif (df.coleta[i] != builder[1]):
            lista.append(tuple(builder.copy()))
            builder = blank.copy()
            builder[0], builder[1] = df.paciente[i], df.coleta[i]
        
        if switcher.get(df.exame[i]) != None:
            builder[switcher.get(df.exame[i])] = df.result[i]

    lista.append(tuple(builder.copy()))
    new_df = pd.DataFrame.from_records(lista, columns=cols)
    new_df.fillna(method='bfill', inplace=True)
    new_df.fillna(method='ffill', inplace=True)
    frames.append(new_df)

    processado_aux = frames[0]
    for i in range(1, len(frames)):
        processado_aux = processado_aux.append(frames[i], ignore_index=True)
    processado = processado_aux.copy(deep=True)

    return processado


def join_data(einstein_data, fleury_data):

    einstein_data['DIM'].replace(['<215', '>7650', '> 7650'], [200, 8000, 8000], inplace=True)
    fleury_data['DIM'].replace('inferior a 215', 200, inplace=True)

    einstein_data['TGP'].replace('<5', '3', inplace=True)

    einstein_data['PCR'].replace(['Detectado', 'Não detectado', 'Inconclusivo'], [True, False, False], inplace=True)
    fleury_data['PCR'].replace(['DETECTADO', 'DETECTADO (POSITIVO)', 'NÃO DETECTADO', 'NÃO DETECTADO (NEGATIVO)'], [True, True, False, False], inplace=True)

    einstein_data['IGG'].replace(['Reagente', 'Não reagente', 'Não Reagente', 'Indeterminado', ' '], [True, False, False, False, False], inplace=True)
    fleury_data['IGG'].replace(['REAGENTE', 'NÃO REAGENTE', 'Indeterminado'], [True, False, False], inplace=True)

    einstein_data['IGM'].replace(['Reagente', 'Reagente Fraco', 'Não reagente', 'Não Reagente', 'Indeterminado', ' '], [True, False, False, False, False, False], inplace=True)
    fleury_data['IGM'].replace(['REAGENTE', 'NÃO REAGENTE', 'Indeterminado'], [True, False, False], inplace=True)

    # Minha hipótese é que campos marcados 'nova coleta' deram resultados que foram descartados
    # e resultaram na requisição de novos exames. Faltam evidências pra confirmar, então os
    # substituirei pelas médias do valor de seus campos.
    nova_coleta = ['nova coleta', 'Nova coleta', 'Nova Coleta', 'NOVA COLETA', 'Nova coleta.']
    dim_data = einstein_data[~einstein_data['DIM'].isin(nova_coleta)]['DIM']
    dhl_data = einstein_data[~einstein_data['DHL'].isin(nova_coleta)]['DHL']
    tgo_data = einstein_data[~einstein_data['TGO'].isin(nova_coleta)]['TGO']
    tgp_data = einstein_data[~einstein_data['TGP'].isin(nova_coleta)]['TGP']
    dim_avg = int(dim_data.astype(int).mean())
    dhl_avg = int(dhl_data.astype(int).mean())
    tgo_avg = int(tgo_data.astype(int).mean())
    tgp_avg = int(tgp_data.astype(int).mean())

    einstein_data['DIM'].replace(nova_coleta, dim_avg, inplace=True)
    einstein_data['DHL'].replace(nova_coleta, dhl_avg, inplace=True)
    einstein_data['TGO'].replace(nova_coleta, tgo_avg, inplace=True)
    einstein_data['TGP'].replace(nova_coleta, tgp_avg, inplace=True)

    einstein_data = einstein_data.append(fleury_data, ignore_index=True)

    return einstein_data.astype({'DIM': 'int32', 'DHL': 'int32', 'TGO': 'int32', 'TGP': 'int32'})


if __name__ == "__main__":

    einstein_data = get_einstein()
    print("Einstein data acquired.")
    fleury_data = get_fleury()
    print("Fleury data acquired.")

    # einstein_report = ProfileReport(einstein_data, title="Einstein report")
    # fleury_report = ProfileReport(fleury_data, title="Fleury report")

    # einstein_report.to_file("einstein_report.html")
    # fleury_report.to_file("fleury_report.html")

    treated_data = join_data(einstein_data, fleury_data)
    print("Data treated successfully.")
    treated_report = ProfileReport(treated_data, title="Treated report")
    treated_report.to_file("treated_data_report.html")
    treated_data.to_csv('treated.csv', index=False, sep='|')