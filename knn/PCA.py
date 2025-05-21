import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def calcular_especificidad(y_true, y_pred, clases):
    especificidades = []
    for clase in clases:
        # Verdaderos negativos
        tn = np.sum((y_true != clase) & (y_pred != clase))
        # Falsos positivos
        fp = np.sum((y_true != clase) & (y_pred == clase))
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        especificidades.append(especificidad)
    return np.mean(especificidades) 

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# Hold-out estratificado 70/30 (semilla 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

# 3. Aplicar PCA (ajustado solo con train)
pca = PCA().fit(X_train)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.9) + 1

print(f"\nNúmero de componentes para 90% de varianza: {n_components}")
print(f"Varianza acumulada: {cumulative_variance[n_components-1]:.2%}")

# Transformar datos
pca = PCA(n_components=n_components).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. Clasificadores
classifiers = {
    "k-NN": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# 5. Evaluación
results = {}
print("\nCOMPARACIÓN DE CLASIFICADORES")
print("="*60)

for name, clf in classifiers.items():
    # Sin PCA
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calcular métricas manualmente
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
    especificidad = calcular_especificidad(y_test, y_pred, np.unique(y))
    balanced_acc = (recall + especificidad) / 2
    
    # Con PCA
    clf.fit(X_train_pca, y_train)
    y_pred_pca = clf.predict(X_test_pca)
    
    precision_pca, recall_pca, _, _ = precision_recall_fscore_support(y_test, y_pred_pca, average='micro')
    especificidad_pca = calcular_especificidad(y_test, y_pred_pca, np.unique(y))
    balanced_acc_pca = (recall_pca + especificidad_pca) / 2
    
    # Guardar resultados
    results[name] = {
        "Sin_PCA": {
            "Sensibilidad": recall,
            "Especificidad": especificidad,
            "Balanced_Accuracy": balanced_acc
        },
        "Con_PCA": {
            "Sensibilidad": recall_pca,
            "Especificidad": especificidad_pca,
            "Balanced_Accuracy": balanced_acc_pca
        },
        "Mejora": balanced_acc_pca - balanced_acc
    }
    
    # Imprimir resultados
    print(f"\n{name}:")
    print("  Sin PCA:")
    print(f"    - Sensibilidad: {recall:.2%}")
    print(f"    - Especificidad: {especificidad:.2%}")
    print(f"    - Balanced Accuracy: {balanced_acc:.2%}")
    print("  Con PCA:")
    print(f"    - Sensibilidad: {recall_pca:.2%}")
    print(f"    - Especificidad: {especificidad_pca:.2%}")
    print(f"    - Balanced Accuracy: {balanced_acc_pca:.2%}")
    print(f"  Diferencia: {results[name]['Mejora']:+.2%}")

# 6. Resumen
mejoras = sum(1 for name in classifiers if results[name]['Mejora'] > 0)
print(f"\nRESUMEN: PCA mejoró {mejoras} de {len(classifiers)} clasificadores")
print(f"Componentes principales usados: {n_components}")