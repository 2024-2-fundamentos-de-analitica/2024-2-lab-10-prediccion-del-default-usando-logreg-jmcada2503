# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import json
import os

train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

train_df = train_df.rename(columns={'default payment next month': 'default'})
test_df = test_df.rename(columns={'default payment next month': 'default'})
train_df.drop(columns=['ID'], inplace=True, errors='ignore')
test_df.drop(columns=['ID'], inplace=True, errors='ignore')
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
train_df = train_df[(train_df["EDUCATION"] != 0) & (train_df["MARRIAGE"] != 0)]
test_df = test_df[(test_df["EDUCATION"] != 0) & (test_df["MARRIAGE"] != 0)]
train_df["EDUCATION"] = train_df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
test_df["EDUCATION"] = test_df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

x_train_data, y_train_data = train_df.drop(columns=['default']), train_df['default']
x_test_data, y_test_data = test_df.drop(columns=['default']), test_df['default']
categorical_cols = x_train_data.select_dtypes(include=['object', 'category']).columns.tolist()

column_transformer = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)],
    remainder='passthrough'
)
pipeline_model = Pipeline([
    ('preprocessor', column_transformer),
    ('scaler', MinMaxScaler()),
    ('feature_selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', LogisticRegression(max_iter=500, random_state=42))
])

hyperparams = {
    'feature_selector__k': range(1, 11),
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'],
    "classifier__max_iter": [100, 200]
}
grid_search_cv = GridSearchCV(pipeline_model, param_grid=hyperparams, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True)
grid_search_cv.fit(x_train_data, y_train_data)

os.makedirs("files/models/", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", 'wb') as file:
    pickle.dump(grid_search_cv, file)

def compute_metrics(y_true, y_pred, dataset):
    return {
        'type': 'metrics',
        'dataset': dataset,
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def confusion_matrix_data(y_true, y_pred, dataset):
    cm_matrix = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": int(cm_matrix[0, 0]), "predicted_1": int(cm_matrix[0, 1])},
        'true_1': {"predicted_0": int(cm_matrix[1, 0]), "predicted_1": int(cm_matrix[1, 1])}
    }

metrics_results = [
    compute_metrics(y_train_data, grid_search_cv.predict(x_train_data), 'train'),
    compute_metrics(y_test_data, grid_search_cv.predict(x_test_data), 'test'),
    confusion_matrix_data(y_train_data, grid_search_cv.predict(x_train_data), 'train'),
    confusion_matrix_data(y_test_data, grid_search_cv.predict(x_test_data), 'test')
]

os.makedirs("files/output/", exist_ok=True)
with open("files/output/metrics.json", "w") as file:
    for result in metrics_results:
        file.write(json.dumps(result) + "\n")
