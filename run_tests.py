# -*- coding: utf-8 -*-
import os
import yaml
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from matplotlib.lines import Line2D

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

# Crear carpeta para figuras (dentro de experiments)
figures_dir = os.path.join(experiments_dir, "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

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
        elif model_name == "LOF":
            models[model_name] = LOF(**params)
        elif model_name == "CBLOF":
            models[model_name] = CBLOF(**params)
        elif model_name == "HBOS":
            models[model_name] = HBOS(**params)
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")
        
    
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
            report_dict = classification_report(y, y_pred, target_names=["inlier", "outlier"], output_dict=True)
            auc_roc = roc_auc_score(y, scores)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

            # Generar visualización con contourf
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            grid = np.c_[xx.ravel(), yy.ravel()]
            try:
                Z = model.decision_function(grid)
            except Exception as e:
                Z = np.zeros(len(grid))
            Z = Z.reshape(xx.shape)
            
            plt.figure()
            cp = plt.contourf(xx, yy, Z, levels=20, cmap='viridis')
            plt.colorbar(cp)
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', marker='o')
            # Obtener el umbral de decisión real basado en el modelo
            if hasattr(model, 'threshold_') and model.threshold_ is not None:
                threshold = model.threshold_
            elif hasattr(model, 'contamination'):
                # Podemos estimar el umbral basado en el percentil de contaminación
                threshold = np.percentile(scores, 100 * (1 - model.contamination))
            else:
                # Fallback a un valor que separe inliers de outliers en los scores
                threshold = np.percentile(scores, 95)  # Asumiendo contamination aprox. 0.05
            plt.contour(xx, yy, Z, levels=[threshold], colors='red', linewidths=0.5)
            plt.title(f"{test_name} - {dataset} - {name} (threshold={threshold:.3f})")

            custom_lines = [Line2D([0], [0], color='blue', marker='o', linestyle='None'),
                            Line2D([0], [0], color='red', marker='o', linestyle='None')]
            plt.legend(custom_lines, ['Inliers', 'Outliers'])
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            
            # Guardar figura
            figure_filename = f"fig_{test_name}_{dataset}_{name}.png"
            figure_filepath = os.path.join(figures_dir, figure_filename)
            plt.savefig(figure_filepath, bbox_inches='tight')

            fig_cm, ax = plt.subplots(figsize=(7, 5))

            # Crear la matriz de confusión con valores absolutos
            cm = confusion_matrix(y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=["Inlier", "Outlier"])
            disp.plot(cmap='Blues', ax=ax, colorbar=False, values_format="d")

            total = cm.sum()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"\n({cm[i, j]/total*100:.1f}%)",
                            ha="center", va="top", color="#0b336d" if cm[i, j] < cm.max()/2 else "white")

            plt.title(f"Matriz de Confusión: {name} en {dataset}\n({n_inliers} inliers, {n_outliers} outliers)", pad=10)
            plt.tight_layout(rect=[0, 0.2, 1, 0.95])  # Hacer espacio para las leyendas
            cm_filename = f"cm_{test_name}_{dataset}_{name}.png"
            cm_filepath = os.path.join(figures_dir, cm_filename)
            plt.savefig(cm_filepath, bbox_inches='tight')
            plt.close('all')
            
            # Preparar un detalle textual (incluir el classification_report completo)
            detalle = classification_report(y, y_pred, target_names=["inlier", "outlier"])
            
            print(f"\n🔍 {name} en dataset '{dataset}' del test '{test_name}'")
            print(f"Tiempo de entrenamiento: {train_time:.2f}s")
            print(detalle)
            print(f"AUC-ROC: {auc_roc:.2f}")
            
            resultados.append({
                "Test": test_name,
                "Dataset": dataset,
                "Modelo": name,
                "TN": tn, 
                "FP": fp, 
                "FN": fn, 
                "TP": tp,
                "Numero atributos": X.shape[1],
                "Tiempo Entrenamiento (s)": round(train_time, 2),
                "Precision Inlier": round(report_dict["inlier"]["precision"], 4),
                "Recall Inlier": round(report_dict["inlier"]["recall"], 4),
                "F1-score Inlier": round(report_dict["inlier"]["f1-score"], 4),
                "Precision Outlier": round(report_dict["outlier"]["precision"], 4),
                "Recall Outlier": round(report_dict["outlier"]["recall"], 4),
                "F1-score Outlier": round(report_dict["outlier"]["f1-score"], 4),
                "AUC-ROC": round(auc_roc, 4),
                "Figura": figure_filepath,
                "Matriz Confusion": cm_filepath,
                "Detalle": detalle
            })
    
    # Guardar resultados de este test en un archivo CSV individual
    df_resultados = pd.DataFrame(resultados)
    filename = f"resultados_{test_name}.csv"
    filepath = os.path.join(experiments_dir, filename)
    df_resultados.to_csv(filepath, index=False)
    print(f"\n✅ Resultados del test '{test_name}' guardados en '{filepath}'")
