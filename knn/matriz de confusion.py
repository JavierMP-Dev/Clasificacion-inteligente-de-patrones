import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                           recall_score, f1_score, balanced_accuracy_score)

# Cargar dataset
wine = load_wine()
X = wine.data
y = wine.target
class_names = wine.target_names

# Inicializar clasificadores
classifiers = {
    'KNeighbors (k=3)': KNeighborsClassifier(n_neighbors=3),
    'GaussianNB': GaussianNB()
}

# Configurar Leave-One-Out
loo = LeaveOneOut()

def calculate_metrics(y_true, y_pred, target_class):
    tn, fp, fn, tp = confusion_matrix(y_true == target_class, y_pred == target_class).ravel()
    return {
        'Class': class_names[target_class],
        'Sensitivity': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'Balanced Accuracy': balanced_accuracy_score(y_true == target_class, y_pred == target_class),
        'Precision': precision_score(y_true == target_class, y_pred == target_class, zero_division=0),
        'F1-Score': f1_score(y_true == target_class, y_pred == target_class, zero_division=0)
    }

def print_confusion_matrix(cm, class_names):
    print("\nMatriz de Confusión:")
    header = " " * 10 + " " * 4 + " ".join(f"{name:^10}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:^10} | " + " ".join(f"{val:^10}" for val in cm[i])
        print(row)

def print_metrics_table(class_metrics):
    print("\n{:<15} {:<12} {:<12} {:<18} {:<12} {:<10}".format(
        "Clase", "Sensibilidad", "Especificidad", "Balanced Accuracy", "Precisión", "F1-Score"))
    for metrics in class_metrics:
        print("{:<15} {:<12.4f} {:<12.4f} {:<18.4f} {:<12.4f} {:<10.4f}".format(
            metrics['Class'], metrics['Sensitivity'], metrics['Specificity'],
            metrics['Balanced Accuracy'], metrics['Precision'], metrics['F1-Score']))

def print_avg_metrics(avg_metrics):
    print("\n{:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Tipo", "Accuracy", "Precision", "Recall", "F1-Score"))
    for avg_type in ['Macro', 'Weighted', 'Micro']:
        print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            avg_type,
            avg_metrics[avg_type]['Accuracy'],
            avg_metrics[avg_type]['Precision'],
            avg_metrics[avg_type]['Recall'],
            avg_metrics[avg_type]['F1']))

# Almacenar resultados
results = {}

for clf_name, clf in classifiers.items():
    y_true_all = []
    y_pred_all = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])
    
    # Matriz de confusión completa
    full_cm = confusion_matrix(y_true_all, y_pred_all)
    
    # Calcular métricas por clase
    class_metrics = []
    for i in range(3):
        metrics = calculate_metrics(np.array(y_true_all), np.array(y_pred_all), i)
        class_metrics.append(metrics)
    
    # Calcular métricas promedio
    avg_metrics = {
        'Macro': {
            'Accuracy': accuracy_score(y_true_all, y_pred_all),
            'Precision': precision_score(y_true_all, y_pred_all, average='macro'),
            'Recall': recall_score(y_true_all, y_pred_all, average='macro'),
            'F1': f1_score(y_true_all, y_pred_all, average='macro')
        },
        'Weighted': {
            'Accuracy': accuracy_score(y_true_all, y_pred_all),
            'Precision': precision_score(y_true_all, y_pred_all, average='weighted'),
            'Recall': recall_score(y_true_all, y_pred_all, average='weighted'),
            'F1': f1_score(y_true_all, y_pred_all, average='weighted')
        },
        'Micro': {
            'Accuracy': accuracy_score(y_true_all, y_pred_all),
            'Precision': precision_score(y_true_all, y_pred_all, average='micro'),
            'Recall': recall_score(y_true_all, y_pred_all, average='micro'),
            'F1': f1_score(y_true_all, y_pred_all, average='micro')
        }
    }
    
    results[clf_name] = {
        'Full Confusion Matrix': full_cm,
        'Class Metrics': class_metrics,
        'Average Metrics': avg_metrics
    }

# Mostrar resultados
for clf_name, result in results.items():
    print(f"\n{'='*50}")
    print(f"Resultados para {clf_name}")
    print('='*50)
    
    print_confusion_matrix(result['Full Confusion Matrix'], class_names)
    print_metrics_table(result['Class Metrics'])
    print_avg_metrics(result['Average Metrics'])