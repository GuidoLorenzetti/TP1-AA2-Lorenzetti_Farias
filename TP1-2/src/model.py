import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import math


def modelo(df):
    """
    Red neuronal de regresión para predecir el rendimiento de los estudiantes.
    """

    #Feature Engineering
    df['aspect_radio'] = df['MajorAxisLength']/df['MinorAxisLength']
    df['roundness'] = (4 * math.pi * df['Area']) / df['Perimeter']**2

    # Convertir las categorías de tipo string a valores numéricos
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])
    
    #Separo las características de la variable objetivo
    X = df.iloc[:, list(range(0, 14)) + list(range(15, 17))].values
    y = df.iloc[:, 14].values

    # Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar características numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Defino modelo
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tf.keras.layers.Dense(units=10, activation="sigmoid"),
    tf.keras.layers.Dense(units=7, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split= 0.2)
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluar el modelo con los datos de prueba
    loss, accuracy = model.evaluate(X_test_scaled, y_test)

    # Imprimir la pérdida y la precisión del modelo en los datos de prueba
    print(f'Loss en datos de prueba: {loss}')
    print(f'Precisión en datos de prueba: {accuracy}')

    sample_index = 2000  # Índice del ejemplo
    sample_example = X_test_scaled[sample_index]  # Tomamos el ejemplo de los datos de prueba

    # Realizar la predicción para el ejemplo de prueba
    prediction = model.predict(sample_example.reshape(1, -1))  # El modelo espera un arreglo 2D, por eso hacemos el reshape

    # La predicción será un arreglo con las probabilidades para cada clase
    predicted_class_index = prediction.argmax()

    # También puedes obtener la clase predicha usando el codificador de etiquetas inverso
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

    # Obtener la clase original del conjunto de datos de prueba
    original_class_index = y_test[sample_index]
    original_class = label_encoder.inverse_transform([original_class_index])[0]

    # Imprimir la clase predicha y la clase original
    print(f'Ejemplo {sample_index + 1}:')
    print(f'  Clase original: {original_class}')
    print(f'  Clase predicha: {predicted_class}')

df = pd.read_csv('TP1-2/src/dataset/Dry-Bean-Dataset.csv')
modelo(df)