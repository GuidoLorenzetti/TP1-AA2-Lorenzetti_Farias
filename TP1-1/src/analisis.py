import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

df = load_csv()
if df is not None:
    process_column(df, "Hours Studied")