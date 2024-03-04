from . import convert_json_to_pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from flask import jsonify


def matrix_conf(y_test, y_pred):

    y_test_array = np.array(y_test)
    confusion_matrix = metrics.confusion_matrix(y_test_array,y_pred)

    # presentar claramente la matriz de confusion  
    # para eso la convertimos a tabla
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    # Desplegamos la matriz de confusión
    cm_display.plot()
    plt.show()

def main():
    df_data = convert_json_to_pd.Convert2DF()
    
    X = df_data.iloc[:, :-1]
    Y = df_data['classification']

    # Dividimos el conjunto dew datos en entrenamiento y prueba 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

    # Paso 3: Preprocesamiento de datos (escalar características)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    svm_model = SVC(kernel='linear', random_state=42)  # Utilizamos un kernel lineal en este ejemplo
    svm_model.fit(X_train, y_train)

    # Paso 5: Hacer predicciones sobre el conjunto de prueba
    y_pred = svm_model.predict(X_test)

    # Paso 6: Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    return jsonify(y_pred)
    
main()