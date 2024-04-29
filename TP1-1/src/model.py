import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def modelo(df):
    """
    Red neuronal de regresión para predecir el rendimiento de los estudiantes.
    """
    # Convertir "Yes" a 1 y "No" a 0
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    X = df.iloc[:, :5].values
    y = df['Performance Index'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar características numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
  
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(units=3, activation = 'sigmoid'),
    tf.keras.layers.Dense(units=1, activation='linear')])
    
    model.compile(optimizer=optimizer, loss='mse', metrics= ['accuracy'])
    
    history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split= 0.2)
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluar el modelo en los datos de prueba
    loss = model.evaluate(X_test, y_test)

    print("Pérdida en los datos de prueba:", loss)

    # Seleccionar una instancia de ejemplo del conjunto de prueba
    ejemplo_idx = 0  # Índice de la instancia de ejemplo que deseas probar
    X_ejemplo = X_test[ejemplo_idx]  # Características del ejemplo seleccionado
    y_true = y_test[ejemplo_idx]  # Etiqueta verdadera del ejemplo seleccionado

    # Realizar una predicción con el modelo
    y_pred = model.predict(X_ejemplo.reshape(1, -1))  # La entrada debe tener forma (1, num_caracteristicas)

    print("Características del ejemplo:", X_ejemplo)
    print("Etiqueta verdadera del ejemplo:", y_true)
    print("Predicción del modelo:", y_pred[0, 0])  # Imprime la predicción

df = pd.read_csv('TP1-1/src/dataset/Student_Performance.csv')
modelo(df)