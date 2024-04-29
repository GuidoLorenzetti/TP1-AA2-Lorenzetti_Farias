import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_csv():
    """
    Carga el archivo CSV con los datos de rendimiento de los estudiantes.
    """
    path = 'TP1-1/src/dataset/Student_Performance.csv'
    if os.path.exists(path):
        performances = pd.read_csv(path)
        return performances
    else:
        return None

def info(df):
    """
    Describe el DataFrame y muestra información sobre los datos faltantes.
    """
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.shape)

    vacios_original=df.isna().sum()
    print(f"Datos vacíos: {vacios_original}")

def plot_variables(df):
    sns.set(style="darkgrid")

    # Obtiene una lista de todas las columnas numéricas del DataFrame
    numeric_columns = df.select_dtypes(include=[float, int]).columns

    # Calcula el número de filas y columnas necesarias para el ploteo
    num_rows = (len(numeric_columns) + 1) // 2
    num_cols = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2 * num_rows, 5 * num_cols))

    # Espacio entre los subplots
    plt.subplots_adjust(wspace=1, hspace=1)

    # Colores distintos para los histogramas
    colors = sns.color_palette("Set2", n_colors=len(numeric_columns))

    # Itera a través de las columnas numéricas y crea un histograma para cada una
    for i, col in enumerate(numeric_columns):
        row = i // num_cols
        col_idx = i % num_cols

        # Selecciona un color diferente para cada histograma
        color = colors[i]

        sns.histplot(data=df, x=col, kde=True, color=color, ax=axs[row, col_idx])

        axs[row, col_idx].set_xlabel(col, fontsize=10)
        axs[row, col_idx].set_ylabel("Count",fontsize=5)

        # Cambia el tamaño de letra de los ejes X y Y
        axs[row, col_idx].tick_params(axis='x', labelsize=5)
        axs[row, col_idx].tick_params(axis='y', labelsize=5)

    # Elimina cualquier subplot no utilizado
    for i in range(len(numeric_columns), num_rows * num_cols):
        row = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axs[row, col_idx])

    
    fig.suptitle(f"Histogramas de variales numéricas", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta el espacio para el título general
    plt.show()

def matriz_correlacion(df):
    corr = df.select_dtypes(include=[float, int]).corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 7))

    ax = sns.heatmap(
        corr,
        #mask = mask,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True,
        annot_kws={'size': 15},
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    # Añade un título general a la figura

    ax.set_title("Matriz de Correlación", fontsize=12)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta el espacio para el título general
    plt.show()

def global_boxplots(df):
    # Asegúrate de que todas las columnas sean numéricas
    columnas_numericas = df.select_dtypes(include=['number'])
    
    if columnas_numericas.empty:
        print("No hay columnas numéricas en el DataFrame.")
        return
    
    # Normalizar las columnas numéricas
    scaler = StandardScaler()
    columnas_numericas_normalized = pd.DataFrame(scaler.fit_transform(columnas_numericas), columns=columnas_numericas.columns)
    
    # Crear una lista de títulos de columnas rotados
    column_titles = [col for col in columnas_numericas_normalized.columns]
    
    # Graficar los boxplots de las columnas numéricas normalizadas con rango en el eje Y de -4 a 4
    plt.figure(figsize=(20, 10))
    plt.title("Boxplots para todas las variables")
    boxplot = plt.boxplot(columnas_numericas_normalized.values, vert=True)
    
    # Rotar los títulos de las columnas
    plt.xticks(range(1, len(columnas_numericas_normalized.columns) + 1), column_titles, rotation=90)
    
    plt.xlabel("Variables")
    plt.ylabel("Valores Normalizados")
    plt.ylim(-4, 4)  # Establecer el rango en el eje Y
    plt.show()


def scatter_plot(df):
    numeric_cols = df.select_dtypes(include=['number'])
    num_plots = len(numeric_cols.columns)
    
    # Determina el número de filas y columnas para la matriz
    num_rows = (num_plots + 2) // 3  # Redondea hacia arriba
    num_cols = min(num_plots, 3)

    # Aumenta el tamaño de la figura
    figsize = (10, num_rows * 3)  # Aumenta la altura de la figura
    
    # Crea una figura de Matplotlib con subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, col in enumerate(numeric_cols.columns):
        row = i // num_cols
        col_num = i % num_cols  # Cambia el nombre de la variable para evitar la confusión

        # Selecciona el subplot actual
        ax = axes[row, col_num]

        # Crea el gráfico de dispersión en el subplot
        sns.scatterplot(x=col, y='Performance Index', data=df, ax=ax)
        ax.set_title(col)  # Configura el título del subplot

    # Elimina subplots no utilizados
    for i in range(num_plots, num_rows * num_cols):
        row = i // num_cols
        col_num = i % num_cols
        fig.delaxes(axes[row, col_num])

    plt.tight_layout()
    plt.show()

def check_balance(df):
    """
    Calcula el balance de clases para las columnas objetivo en un DataFrame.
    """
    # Definir los intervalos y calcular el balance
    intervalos = [(10, 30), (31, 50), (51, 70), (71, 90), (91, 100)]
    total_valores = len(df['Performance Index'])

    conteos = {}
    for intervalo in intervalos:
        inicio, fin = intervalo
        conteo = ((df['Performance Index'] >= inicio) & (df['Performance Index'] <= fin)).sum()
        conteos[intervalo] = conteo

    balance = {intervalo: (conteo / total_valores) * 100 for intervalo, conteo in conteos.items()}

    # Crear un DataFrame para el gráfico
    nombres_intervalos = [f"({inicio}-{fin})" for inicio, fin in intervalos]
    data = pd.DataFrame({'Intervalo': nombres_intervalos, 'Balance': list(balance.values())})

    # Crear el gráfico de barras utilizando Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Intervalo', y='Balance', data=data, color='skyblue')
    plt.title('Balance de la columna "Performance Index"')
    plt.xlabel('Intervalo')
    plt.ylabel('Porcentaje de valores')

    # Mostrar los porcentajes en las barras
    for index, value in enumerate(balance.values()):
        plt.text(index, value + 1, f'{value:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45)  # Rotar los nombres de los intervalos para mejor legibilidad
    plt.ylim(0, 100)  # Establecer el rango del eje y de 0 a 100
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Agregar una cuadrícula en el eje y
    plt.tight_layout()  # Ajustar el diseño para evitar superposiciones
    plt.show()

def graficos(df):
    """
    Graficos de variables numéricas y matriz de correlación
    """
    # Imprimir la cantidad de datos disponibles
    print(f"Cantidad de datos disponibles en el DataFrame: {len(df)}")
    plot_variables(df)
    matriz_correlacion(df)
    global_boxplots(df)
    scatter_plot(df)
    check_balance(df)


info(load_csv())
graficos(load_csv())