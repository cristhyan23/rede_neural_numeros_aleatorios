import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import date,datetime
import tensorflow as tf


#VALIDADOR ACERTOS
acertos_quadra = 0
num_acerto_quadra = []
acertos_quina = 0
num_acerto_quina = []
acertos_sexta = 0
num_acerto_sexta = []

#variavel para conferir acertos
aprendizado = True
rodadas = 0

def adiciona_data_frame(palpite, data, acerto):
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
                               'Acertou': [acerto]})
    # Concatenar o DataFrame existente com a nova linha
    data = pd.concat([data, nova_linha], ignore_index=True)
    # Salvar o DataFrame atualizado de volta para o Excel
    data.to_csv('./sorteios_mega_sena.csv', index=False)
    return data

# Loop de aprendizado
while aprendizado:
    # Abrir o arquivo Excel
    data = pd.read_csv("./sorteios_mega_sena.csv")

    numeros_certos = 0

    # Dividir os dados em características (X) e rótulos (y)
    X = data.drop(columns=["Data","Acertou"])  # Características
    y = data["Acertou"]  # Rótulos

    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertendo os dados para numpy arrays
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_train_np = X_train_np.astype('float32')
    y_train_np = y_train_np.astype('float32')

    # Criar uma pasta de log para o TensorBoard
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")


    # Construindo o modelo da rede neural
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_np.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compilando o modelo
    model_nn.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar o modelo com o callback do TensorBoard
    model_nn.fit(X_train_np, y_train_np, epochs=10, batch_size=32, verbose=0)

    # Gerar Sorteio dos 6 números
    sorteio_mega = np.random.choice(range(1, 61), size=6, replace=False)  # Selecionar 6 números aleatórios

    # Adicionar uma característica arbitrária
    media_sorteados = np.mean(sorteio_mega)
    sorteio_mega_com_media = np.append(sorteio_mega, media_sorteados)

    # Realizar previsão com a rede neural
    next_draws_nn = model_nn.predict(np.array([sorteio_mega_com_media]))

    if next_draws_nn == 1:
        palpite = sorteio_mega
    else:
        palpite = list(np.random.choice(range(1, 61), size=6, replace=False))  # Gerar palpite aleatório

    for num in sorteio_mega:
        if num in palpite:  # Verificar se o número está presente no palpite
            numeros_certos += 1

    if numeros_certos == 4:
        print('Parabens Você acertou a Quadra')
        acertos_quadra += 1
        data = adiciona_data_frame(palpite, data, 1)
    elif numeros_certos == 5:
        acertos_quina += 1
        print('Parabens você acertou a quina')
        data = adiciona_data_frame(palpite, data, 1)
    elif numeros_certos == 6:
        print('Parabens você acertou a sexta')
        data = adiciona_data_frame(palpite, data, 1)
        acertos_sexta += 1
    else:
        print(f"Que pena você perdeu acertou somente:{numeros_certos} \n Palpite: {palpite} Sorteio: {sorteio_mega}")
        data = adiciona_data_frame(palpite, data, 0)

    rodadas += 1
    resultado_sexta = acertos_sexta/rodadas * 100
    if resultado_sexta >= 1:
        aprendizado = False


print(f'--------RESULTADO FINAL---Rodadas: {rodadas}------')
print(f'Acertos Quadra {round(acertos_quadra/rodadas*100,2)}%')
print(f'Numero acertos quadra {num_acerto_quadra}')
print('----------------------------------------------------------------')
print(f'Acertos Quina {round(acertos_quina/rodadas*100,2)}%')
print(f'Numero acertos Quina {num_acerto_quina}')
print('----------------------------------------------------------------')
print(f'Acertos Sexta {round(acertos_sexta/rodadas*100,2)}%')
print(f'Numero acertos Sexta {num_acerto_sexta}')