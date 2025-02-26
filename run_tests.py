# -*- coding: utf-8 -*-
import os
import yaml
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

# Cargar configuración desde config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def generar_datos(tipo, n_inliers, n_outliers, random_state=42):
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
    
    # Combinar y escalar (usando solo los inliers para el escalado)
    X = np.vstack([X_inliers, X_outliers])
    y = np.hstack([y_inliers, y_outliers])
    scaler = StandardScaler().fit(X_inliers)
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

# Crear carpeta para experimentos si no existe
experiments_dir = "experiments"
if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)

# Iterar sobre cada test configurado en el YAML
for test in config.get("tests", []):
    test_name = test.get("name", "test_unnamed")
    n_inliers = test.get("n_inliers", 950)
    n_outliers = test.get("n_outliers", 50)
    datasets_config = test.get("datasets", ["moons", "circles", "blobs"])
    models_config = test.get("models", {})

    # Configurar los modelos a partir del YAML
    models = {}
    for model_name, params in models_config.items():
        if model_name == "IForest":
            models[model_name] = IForest(**params)
        elif model_name == "KNN":
            models[model_name] = KNN(**params)
        elif model_name == "OCSVM":
            models[model_name] = OCSVM(**params)
        elif model_name == "COPOD":
            models[model_name] = COPOD(**params)
        elif model_name == "ECOD":
            models[model_name] = ECOD(**params)
    
    resultados = []  # Lista para almacenar los resultados de este test

    for dataset in datasets_config:
        print(f"\n=== Test '{test_name}' - Evaluando dataset: {dataset} ===")
        X, y = generar_datos(dataset, n_inliers, n_outliers)
        
        for name, model in models.items():
            # Entrenar solo con inliers
            start_time = time.time()
            model.fit(X[y == 0])
            train_time = time.time() - start_time
            
            # Predecir en todo el dataset
            y_pred = model.predict(X)
            scores = model.decision_function(X)
            
            # Generar reporte de clasificación (output_dict para extraer métricas)
            report = classification_report(y, y_pred, target_names=["inlier", "outlier"], output_dict=True)
            auc_roc = roc_auc_score(y, scores)
            
            print(f"\n🔍 {name} en dataset '{dataset}' del test '{test_name}'")
            print(f"Tiempo de entrenamiento: {train_time:.2f}s")
            print(classification_report(y, y_pred, target_names=["inlier", "outlier"]))
            print(f"AUC-ROC: {auc_roc:.2f}")
            
            resultados.append({
                "Test": test_name,
                "Dataset": dataset,
                "Modelo": name,
                "Stats": f"{n_inliers} - {n_outliers} - {X.shape[1]}",
                "T. Entreno (s)": round(train_time, 4),
                "Precision In.": round(report["inlier"]["precision"], 4),
                "Recall In.": round(report["inlier"]["recall"], 4),
                "F1-score In.": round(report["inlier"]["f1-score"], 4),
                "Precision Out.": round(report["outlier"]["precision"], 4),
                "Recall Out.": round(report["outlier"]["recall"], 4),
                "F1-score Out.": round(report["outlier"]["f1-score"], 4),
                "AUC-ROC": round(auc_roc, 4)
            })
    
    # Guardar resultados de este test en un archivo CSV individual
    df_resultados = pd.DataFrame(resultados)
    filename = f"resultados_{test_name}.csv"
    filepath = os.path.join(experiments_dir, filename)
    df_resultados.to_csv(filepath, index=False)
    print(f"\n✅ Resultados del test '{test_name}' guardados en '{filepath}'")
