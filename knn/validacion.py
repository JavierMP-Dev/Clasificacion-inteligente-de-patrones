from sklearn.datasets import load_wine
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# Cargar el dataset Wine
wine = load_wine()
X = wine.data  # Características
y = wine.target  # Etiquetas

# Ver información básica del dataset
print(f"Número de muestras: {X.shape[0]}")
print(f"Número de características: {X.shape[1]}")
print(f"Nombres de las características: {wine.feature_names}")
print(f"Clases: {wine.target_names}")


# Crear el objeto LeaveOneOut
loo = LeaveOneOut()

# Modelo a evaluar (puedes cambiarlo por el que necesites)
model = RandomForestClassifier(random_state=42)

# Lista para almacenar los resultados
accuracies = []

# Iterar sobre las divisiones train-test
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Entrenar y predecir
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcular precisión para esta iteración
    accuracies.append(accuracy_score(y_test, y_pred))

# Calcular métricas finales
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\nResultados de la validación Leave-One-Out:")
print(f"Precisión promedio: {mean_accuracy:.4f}")
print(f"Desviación estándar: {std_accuracy:.4f}")
print(f"Número de iteraciones: {len(accuracies)}")

from sklearn.metrics import confusion_matrix

# Matriz de confusión acumulada
y_true_all = []
y_pred_all = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_true_all.append(y_test[0])
    y_pred_all.append(y_pred[0])

# Mostrar matriz de confusión
conf_mat = confusion_matrix(y_true_all, y_pred_all)
print("\nMatriz de confusión:")
print(conf_mat)