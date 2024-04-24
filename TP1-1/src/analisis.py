import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

def load_csv():
    path = 'TP1-1/src/dataset/Student_Performance.csv'
    if os.path.exists(path):
        performances = pd.read_csv(path)
        return performances
    else:
        return None

def process_column(df, column_name):
    column = df[column_name]
    column_info = column.describe()
    column_count = column.value_counts()
    
    if column.dtype == np.float64 or column.dtype == np.int64:
        column_dict = {
            'name': column_name,
            'mean': column_info['mean'],
            'std': column_info['std'],
            'min': column_info['min'],
            '25%': column_info['25%'],
            '50%': column_info['50%'],
            '75%': column_info['75%'],
            'max': column_info['max'],
            'count': column_info['count'],
            'mode': column_count.idxmax(),
            'mode_count': column_count.max()
        }
    
    else:
        column_dict = {
            'name': column_name,
            'count': column_info['count'],
            'unique': column_info['unique'],
            'top': column_info['top'],
            'freq': column_info['freq']
        }

    for key in column_dict:
        print(f"{key:<10}\t{column_dict[key]}")

    return column_dict

df = load_csv()
columns_info = []

if df is not None:

    for column in df.columns:
        column_dict = process_column(df, column)
        columns_info.append(column_dict)