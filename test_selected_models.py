# -*- coding: utf-8 -*-
import numpy as np
import time
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD

def generar_datos(tipo, n_inliers=950, n_outliers=50, random_state=42):
    """
    Genera un dataset sintético con inliers y outliers.
    - tipo: 'moons', 'circles' o 'blobs'
    """
    np.random.seed(random_state)
    
    if tipo == 'moons':
        X_inliers, _ = make_moons(n_samples=n_inliers, noise=0.05, random_state=random_state)
    elif tipo == 'circles':
        X_inliers, _ = make_circles(n_samples=n_inliers, noise=0.05, factor=0.5, random_state=random_state)
    elif tipo == 'blobs':
        X_inliers, _ = make_blobs(n_samples=n_inliers, centers=1, cluster_std=1.0, random_state=random_state)
    else:
        raise ValueError("Tipo de datos desconocido. Usa 'moons', 'circles' o 'blobs'.")
    
    # Todos los inliers tienen etiqueta 0
    y_inliers = np.zeros(n_inliers)
    
    # Generar outliers de forma uniforme
    X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, X_inliers.shape[1]))
    y_outliers = np.ones(n_outliers)
    
    # Combinar
    X = np.vstack([X_inliers, X_outliers])
    y = np.hstack([y_inliers, y_outliers])
    
    # Escalar usando solo los inliers (simula el entorno real)
    scaler = StandardScaler().fit(X_inliers)
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

# Definir modelos de outlier detection
models = {
    "IForest": IForest(n_estimators=300, max_samples=64, contamination=0.05, random_state=42),
    "KNN": KNN(n_neighbors=3, method="largest", contamination=0.05),
    "OCSVM": OCSVM(kernel="rbf", nu=0.05, gamma=0.1, contamination=0.05),
    "COPOD": COPOD(contamination=0.05),
    "ECOD": ECOD(contamination=0.05)
}

# Lista de tipos de datasets a evaluar
tipos = ['moons', 'circles', 'blobs']
n_inliers = 95000
n_outliers = 5000

# Lista para almacenar los resultados
resultados = []

for tipo in tipos:
    print(f"\n=== Evaluando dataset: {tipo} ===")
    X, y = generar_datos(tipo, n_inliers, n_outliers)
    
    for name, model in models.items():
        # Entrenar solo con inliers
        start_time = time.time()
        model.fit(X[y == 0])
        train_time = time.time() - start_time
        
        # Predecir sobre todo el conjunto
        y_pred = model.predict(X)
        scores = model.decision_function(X)
        
        report = classification_report(y, y_pred, target_names=["inlier", "outlier"], output_dict=True)
        auc_roc = roc_auc_score(y, scores)

        print(f"\n🔍 {name} en dataset '{tipo}'")
        print(f"Tiempo de entrenamiento: {train_time:.2f}s")
        print(classification_report(y, y_pred, target_names=["inlier", "outlier"]))
        print(f"AUC-ROC: {auc_roc:.2f}")

        # Guardar resultados en la lista
        resultados.append({
            "Dataset": tipo,
            "Modelo": name,
            "Tiempo Entrenamiento (s)": round(train_time, 2),
            "Precision Inlier": round(report["inlier"]["precision"], 2),
            "Recall Inlier": round(report["inlier"]["recall"], 2),
            "F1-score Inlier": round(report["inlier"]["f1-score"], 2),
            "Precision Outlier": round(report["outlier"]["precision"], 2),
            "Recall Outlier": round(report["outlier"]["recall"], 2),
            "F1-score Outlier": round(report["outlier"]["f1-score"], 2),
            "AUC-ROC": round(auc_roc, 2)
        })

# Convertir a DataFrame y guardar en Excel
df_resultados = pd.DataFrame(resultados)
nombre_fichero = f"resultados_{n_inliers}-inliners_{n_outliers}-outliers.xlsx"
df_resultados.to_excel(nombre_fichero, index=False)

print(f"\n✅ Resultados guardados en '{nombre_fichero}'")
