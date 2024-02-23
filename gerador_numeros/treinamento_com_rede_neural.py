import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
import signal
import sys

# Função para tratamento do sinal de interrupção (CTRL+C)
def handle_ctrl_c(signal, frame):
    print("\nEncerrando aplicação...")
    # Encerrar a aplicação
    sys.exit(0)

# Registrar o tratamento do sinal de interrupção (CTRL+C)
signal.signal(signal.SIGINT, handle_ctrl_c)

# Loop infinito para manter o programa em execução até que o sinal de interrupção seja recebido
print("Pressione CTRL+C para analisar os dados...")


#MODELO QUE SALVA O RESULTADO DENTRO DO ARQUIVO PARA APRENDIZADO
def adiciona_data_frame(palpite, data, acerto,bolas_acerto):
    # Encontrar o maior valor na coluna "Concurso" e incrementar
    maior_concurso = data['Concurso'].max()
    novo_concurso = maior_concurso + 1
    # Criar um novo DataFrame com os valores da nova linha
    nova_linha = pd.DataFrame({'Concurso': [novo_concurso], 'Data': [datetime.today().strftime('%d/%m/%Y')],
                               'bola 1': [palpite[0]],
                               'bola 2': [palpite[1]],
                               'bola 3': [palpite[2]],
                               'bola 4': [palpite[3]],
                               'bola 5': [palpite[4]],
                               'bola 6': [palpite[5]],
                               'Acertou': [acerto],
                               'acertobola1':[bolas_acerto[0]],
                               'acertobola2':[bolas_acerto[1]],
                               'acertobola3':[bolas_acerto[2]],
                               'acertobola4':[bolas_acerto[3]],
                               'acertobola5':[bolas_acerto[4]],
                               'acertobola6':[bolas_acerto[5]]})
    # Concatenar o DataFrame existente com a nova linha
    data = pd.concat([data, nova_linha], ignore_index=True)
    # Salvar o DataFrame atualizado de volta para o Excel
    data.to_csv('./sorteios_mega_sena.csv', index=False)
    return data


#FUNÇÃO DE APRENDIZADO DA REDE NEURAL
def modelo_rede_neural():
    # Abrir o arquivo Excel
    data = pd.read_csv("./sorteios_mega_sena.csv")


    # Dividir os dados em características (X) e rótulos (y)
    X = data.drop(columns=["Data", "Acertou", "acertobola1", "acertobola2", "acertobola3", "acertobola4", "acertobola5",
                           "acertobola6"])  # Características
    y = data[["Acertou", "acertobola1", "acertobola2", "acertobola3", "acertobola4", "acertobola5",
              "acertobola6"]]  # Rótulos

    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertendo os dados para numpy arrays
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_train_np = X_train_np.astype('float32')
    y_train_np = y_train_np.astype('float32')

    # Construindo o modelo da rede neural
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_np.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compilando o modelo
    model_nn.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar o modelo
    model_nn.fit(X_train_np, y_train_np, epochs=10, batch_size=32, verbose=0)

    return model_nn,data


#VALIDADOR ACERTOS
acertos_quadra = 0
num_acerto_quadra = []
acertos_quina = 0
num_acerto_quina = []
acertos_sexta = 0
num_acerto_sexta = []
acertou_num = 1
errou_num = 0
#variavel para conferir acertos
aprendizado = True
rodadas = 0
# Loop de aprendizado
while aprendizado:
#CONTADOR DE ACERTOS
    numeros_certos = 0

    # Gerar Sorteio dos 6 números
    sorteio_mega = np.random.choice(range(1, 61), size=6, replace=False)  # Selecionar 6 números aleatórios

    # Adicionar uma característica arbitrária
    media_sorteados = np.mean(sorteio_mega)
    sorteio_mega_com_media = np.append(sorteio_mega, media_sorteados)

    #recebe dados gerando dentro do modelo nueral
    modelo,data = modelo_rede_neural()
    # Realizar previsão com a rede neural
    next_draws_nn = modelo.predict(np.array([sorteio_mega_com_media]))
    print(f'Essa foi a previsão gerada pelo modelo: {next_draws_nn}')

    if next_draws_nn > 0.99999999999998:
        palpite = sorteio_mega
    else:
        palpite = list(np.random.choice(range(1, 61), size=6, replace=False))  # Gerar palpite aleatório

    bolas_acerto = []
    for num in sorteio_mega:
        if num in palpite:  # Verificar se o número está presente no palpite
            numeros_certos += 1
            bolas_acerto.append(acertou_num)
        else:
            bolas_acerto.append(errou_num)

    if numeros_certos in (4, 5, 6):
        if numeros_certos == 4:
            print('Parabens Você acertou a Quadra')
            acertos_quadra += 1
        elif numeros_certos == 5:
            acertos_quina += 1
            print('Parabens você acertou a quina')
        else:  # Quando numeros_certos == 6
            print('Parabens você acertou a sexta')
            acertos_sexta += 1
        data = adiciona_data_frame(palpite, data, 1, bolas_acerto)
    else:
        print(f"Que pena você perdeu acertou somente:{numeros_certos} \n Palpite: {palpite} Sorteio: {sorteio_mega}")
        data = adiciona_data_frame(palpite, data, 0, bolas_acerto)

    rodadas += 1
    resultado_sexta = acertos_sexta/rodadas * 100
    if acertos_sexta >= 100:
        print(f'Acertos da Sexta:{round(resultado_sexta,2)}%')
        aprendizado = False
