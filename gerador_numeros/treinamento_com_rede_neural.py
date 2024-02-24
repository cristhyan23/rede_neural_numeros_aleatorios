import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

# Load the Mega Sena data
data = pd.read_csv("./sorteios_mega_sena.csv", parse_dates=["Data"], dayfirst=True)

# Columns for drawn numbers
drawn_numbers_cols = ["bola 1", "bola 2", "bola 3", "bola 4", "bola 5", "bola 6"]



def plot_learning_curves(history):
    # Obtém as métricas de treinamento
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']

    # Obtém as métricas de validação (se disponíveis)
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']
        has_validation = True
    else:
        has_validation = False

    # Cria o gráfico
    plt.figure(figsize=(12, 6))

    # Plota a perda de treinamento
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    if has_validation:
        plt.plot(val_loss, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plota a acurácia de treinamento
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    if has_validation:
        plt.plot(val_accuracy, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Exibe o gráfico
    plt.tight_layout()
    plt.show()


def has_duplicates(lst):
        # Convertendo arrays numpy em tuplas
        lst_tuples = [tuple(arr) for arr in lst]
        counts = Counter(lst_tuples)
        for count in counts.values():
            if count > 1:
                return True
        return False

# Function to calculate features
def calculate_features(data):
    # Differences between consecutive draws
    for i in range(1, len(drawn_numbers_cols)):
        col_name = f'diff_{i}'
        data[col_name] = data[drawn_numbers_cols].diff(axis=1).iloc[:, i]

    # Number frequencies (last 100 draws)
    for num in range(1, 61):
        col_name = f'freq_{num}'
        data[col_name] = data[drawn_numbers_cols].apply(lambda row: (row == num).sum(), axis=1).rolling(500).mean()

    # Convert the 'Data' column to datetime
    data['Data'] = pd.to_datetime(data['Data'], format='%d/%m/%Y')

    # Day of the week
    data['day_of_week'] = data['Data'].dt.dayofweek

    return data

# Function to preprocess data for the neural network
def preprocess_data(data):
    # Restructure data: Use features from draw 'n' to predict draw 'n + 1'
    X = data.drop(columns=["Concurso", "Data", "Acertou", "AcertosIndividuais"],axis=1)
    y = data["Acertou"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the neural network model
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='linear'))  # Output 6 numbers (consider rounding)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


# Function to train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Function to generate predictions
def generate_predictions(model, X_test):
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    rounded_predictions = np.round(predictions).astype(int)
    return predictions.reshape(-1, 6)


# Function to analyze a single draw against predictions
def analyze_draw(data, draw, predictions):
    matches = np.all(draw == predictions[0])  # Remova axis=1, pois estamos lidando com um único conjunto de números
    num_matches = np.sum(draw == predictions[0])  # Remova axis=1, pois estamos lidando com um único conjunto de números
    new_row = pd.DataFrame({'Concurso': [data['Concurso'].max() + 1],
                            'Data': [pd.to_datetime('today').strftime('%d/%m/%Y')],
                            **{f'bola {i + 1}': [int(predictions[0][i])] for i in range(6)},
                            'Acertou': int(matches.any()),
                            'AcertosIndividuais': num_matches})
    # Use pd.concat to add the new row
    return pd.concat([data, new_row], ignore_index=True)

nao_acertou_6_numeros = True
while nao_acertou_6_numeros:

    # Preprocess the data (calculate features first)
    data = calculate_features(data.copy())
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Create the model
    model = create_model(X_train.shape[1:])

    # Train the model
    train_model(model, X_train, y_train)

    # Converter os valores para o formato desejado ("%d/%m/%Y")
    data['Data'] = pd.to_datetime(data['Data'], format='%d-%m-%Y')

    #Main Loop
    for _ in range(10):

        draw = np.random.choice(range(1, 61), size=6, replace=False)
        # Initialize a set to keep track of unique numbers in predictions
        predictions = generate_predictions(model, X_test.iloc[-1].to_numpy().reshape(1, -1))
        # Multiply each predicted number by the corresponding drawn number
        predictions = np.round(predictions * draw).astype(int)
        while has_duplicates(predictions):
            predictions = generate_predictions(model, X_test.iloc[-1].to_numpy().reshape(1, -1))
            # Multiply each predicted number by the corresponding drawn number
            predictions = np.round(predictions * draw).astype(int)

        # Convert the unique numbers into a list
        final_predictions = sorted(list(predictions))

        # Analyze draw with the final predictions
        data = analyze_draw(data, draw, final_predictions)


        # Print results
        print(f"\nDraw: {draw}")
        print(f"Prediction: {final_predictions[0]}")
        print(f"Exact Matches: {np.sum(draw == final_predictions)}")  # Print number of exact matches
        #VALIDA SE ACERTOU OS 6 NÚMEROS
        if np.sum(draw == final_predictions) == 6:
            nao_acertou_6_numeros = False

    # Salvar o DataFrame no arquivo CSV
    data.to_csv("./sorteios_mega_sena.csv", index=False)


# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_train, y_train))
# Plotar curvas de aprendizado
plot_learning_curves(history)
