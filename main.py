#!/usr/bin/env python3
"""
Punto de entrada principal para el sistema de detección de anomalías.

Este script proporciona una interfaz de línea de comandos para interactuar con
el sistema de detección de anomalías, así como ejemplos de uso para diferentes
escenarios (datos sintéticos, procesamiento por lotes, tiempo real, etc.)
"""

import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple

# Importar componentes del sistema
from config import load_config_from_json
from anomaly_detector import AnomalyDetector, AnomalyDetectionResult

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100


def generate_synthetic_data(
    n_normal: int = 1000,
    n_anomalies: int = 50,
    n_features: int = 5,
    contamination: float = 0.05,
    noise_level: float = 0.1,
    regime_changes: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Genera datos sintéticos para demostración del sistema.
    
    Args:
        n_normal: Número de puntos normales
        n_anomalies: Número de anomalías
        n_features: Número de características
        contamination: Proporción de anomalías
        noise_level: Nivel de ruido en los datos
        regime_changes: Si generar cambios de régimen
        seed: Semilla para reproducibilidad
        
    Returns:
        Tupla con (datos, etiquetas, nombres de características)
    """
    np.random.seed(seed)
    logger.info(f"Generando {n_normal} puntos normales y {n_anomalies} anomalías con {n_features} características")
    
    # Crear nombres de características
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Generar datos normales
    if regime_changes:
        # Crear diferentes regímenes
        n_regimes = 3
        points_per_regime = n_normal // n_regimes
        
        normal_data = []
        for i in range(n_regimes):
            # Cada régimen tiene una media diferente
            regime_mean = np.random.uniform(-1, 1, n_features) * (i + 1)
            regime_data = np.random.normal(
                regime_mean, 
                1.0, 
                (points_per_regime, n_features)
            )
            normal_data.append(regime_data)
            
        normal_data = np.vstack(normal_data)
        # Añadir ruido para evitar límites perfectos entre regímenes
        normal_data += np.random.normal(0, noise_level, normal_data.shape)
    else:
        # Sin cambios de régimen, solo datos normales centrados en cero
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
    
    # Handle special case: no anomalies requested
    if n_anomalies == 0:
        # Return only normal data with appropriate labels
        all_data = normal_data
        labels = np.ones(all_data.shape[0])
        return all_data, labels, feature_names
    
    # Generar anomalías (desplazadas respecto al centro)
    anomalies = []
    for _ in range(n_anomalies):
        # Diferentes tipos de anomalías
        anomaly_type = np.random.randint(0, 3)
        
        if anomaly_type == 0:
            # Anomalía global: desviación en todas las características
            anomaly = np.random.normal(5, 1, n_features)
        elif anomaly_type == 1:
            # Anomalía en características específicas
            anomaly = np.random.normal(0, 1, n_features)
            # Seleccionar 2 características al azar para hacerlas anómalas
            anomalous_features = np.random.choice(n_features, 2, replace=False)
            anomaly[anomalous_features] = np.random.normal(5, 1, 2)
        else:
            # Anomalía de correlación
            anomaly = np.random.normal(0, 1, n_features)
            # Hacer que 2 características estén correlacionadas anormalmente
            anomalous_features = np.random.choice(n_features, 2, replace=False)
            anomaly[anomalous_features[0]] = anomaly[anomalous_features[1]] * -3
            
        anomalies.append(anomaly)
    
    anomaly_data = np.array(anomalies)
    
    # Mezclar datos normales y anómalos
    all_data = np.vstack([normal_data, anomaly_data])
    
    # Crear etiquetas (1 = normal, -1 = anomalía)
    labels = np.ones(all_data.shape[0])
    labels[n_normal:] = -1
    
    # Aleatorizar orden
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    labels = labels[indices]
    
    logger.info(f"Datos sintéticos generados: {all_data.shape[0]} puntos, {np.sum(labels == -1)} anomalías")
    
    return all_data, labels, feature_names


def generate_time_series_data(
    n_points: int = 1000,
    n_features: int = 5,
    n_anomalies: int = 50,
    noise_level: float = 0.1,
    seasonal_patterns: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Genera datos sintéticos de series temporales.
    
    Args:
        n_points: Número total de puntos
        n_features: Número de características
        n_anomalies: Número de anomalías
        noise_level: Nivel de ruido en los datos
        seasonal_patterns: Si generar patrones estacionales
        seed: Semilla para reproducibilidad
        
    Returns:
        Tupla con (datos, etiquetas, nombres de características)
    """
    np.random.seed(seed)
    logger.info(f"Generando serie temporal con {n_points} puntos y {n_features} características")
    
    # Crear nombres de características
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Generar datos de serie temporal
    time_idx = np.linspace(0, 10 * np.pi, n_points)
    data = np.zeros((n_points, n_features))
    
    # Componente base para cada característica
    for i in range(n_features):
        if seasonal_patterns:
            # Cada característica tiene un patrón estacional diferente
            frequency = 1 + i * 0.2
            amplitude = 1 + i * 0.5
            phase = i * np.pi / n_features
            
            data[:, i] = amplitude * np.sin(frequency * time_idx + phase)
            
            # Añadir tendencia
            data[:, i] += np.linspace(0, i * 0.5, n_points)
        else:
            # Sin patrones estacionales, solo procesos aleatorios
            data[:, i] = np.random.normal(0, 1, n_points)
            # Hacer que sea un paseo aleatorio
            data[:, i] = np.cumsum(data[:, i]) * 0.1
    
    # Añadir ruido
    data += np.random.normal(0, noise_level, data.shape)
    
    # Generar etiquetas (1 = normal, -1 = anomalía)
    labels = np.ones(n_points)
    
    # Insertar anomalías
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Diferentes tipos de anomalías
        anomaly_type = np.random.randint(0, 3)
        
        if anomaly_type == 0:
            # Anomalía de pico (spike)
            data[idx, :] += np.random.normal(5, 1, n_features)
        elif anomaly_type == 1:
            # Anomalía en características específicas
            anomalous_features = np.random.choice(n_features, 2, replace=False)
            data[idx, anomalous_features] += np.random.normal(5, 1, 2)
        else:
            # Cambio de nivel
            if idx < n_points - 10:
                # Afecta a los próximos 5-10 puntos
                length = np.random.randint(5, 10)
                affected_feature = np.random.randint(0, n_features)
                data[idx:idx+length, affected_feature] += 3
        
        labels[idx] = -1
    
    logger.info(f"Serie temporal generada: {n_points} puntos, {np.sum(labels == -1)} anomalías")
    
    return data, labels, feature_names


def save_results(results: List[AnomalyDetectionResult], filename: str) -> None:
    """
    Guarda los resultados de detección en un archivo JSON.
    
    Args:
        results: Lista de resultados de detección
        filename: Nombre del archivo para guardado
    """
    # Convertir resultados a formato serializable
    serializable_results = []
    for result in results:
        result_dict = result.to_dict()
        # Process all values recursively to ensure JSON serializability
        serializable_results.append(convert_to_serializable(result_dict))
    
    # Guardar en JSON con formato legible
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Resultados guardados en {filename}")


def convert_to_serializable(obj):
    """
    Convierte recursivamente cualquier objeto a un formato serializable en JSON.
    
    Args:
        obj: Objeto a convertir
        
    Returns:
        Versión serializable del objeto
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For any other type, convert to string
        return str(obj)


def demo_batch_processing() -> None:
    """
    Demostración de detección de anomalías por lotes.
    """
    logger.info("Iniciando demostración de procesamiento por lotes")
    
    # Generar datos sintéticos
    data, labels, feature_names = generate_synthetic_data(
        n_normal=1000,
        n_anomalies=50,
        n_features=8,
        contamination=0.05,
        regime_changes=True
    )
    
    # Dividir en entrenamiento (solo normales) y test
    train_mask = labels == 1
    train_ratio = 0.7
    train_indices = np.where(train_mask)[0]
    np.random.shuffle(train_indices)
    n_train = int(len(train_indices) * train_ratio)
    
    train_data = data[train_indices[:n_train]]
    test_data = data[train_indices[n_train:]]
    test_labels = labels[train_indices[n_train:]]
    
    # Añadir anomalías al conjunto de test
    anomaly_indices = np.where(labels == -1)[0]
    test_data = np.vstack([test_data, data[anomaly_indices]])
    test_labels = np.hstack([test_labels, labels[anomaly_indices]])
    
    # Inicializar detector
    logger.info("Inicializando detector de anomalías")
    detector = AnomalyDetector()
    
    # Definir callback para anomalías
    def anomaly_alert(result: AnomalyDetectionResult) -> None:
        if result.is_anomaly:
            logger.info(f"¡ALERTA! Anomalía detectada: {result}")
    
    detector.add_anomaly_callback(anomaly_alert)
    
    # Entrenar detector
    logger.info("Entrenando detector con datos normales")
    detector.fit(train_data, feature_names=feature_names)
    
    # Procesar lote de datos
    logger.info(f"Procesando lote de {len(test_data)} puntos")
    start_time = time.time()
    results = detector.process_batch(test_data, explain_anomalies=True)
    processing_time = time.time() - start_time
    
    # Analizar resultados
    predictions = [1 if not r.is_anomaly else -1 for r in results]
    true_anomalies = np.sum(test_labels == -1)
    detected_anomalies = np.sum(np.array(predictions) == -1)
    
    logger.info(f"Procesamiento completado en {processing_time:.2f} segundos")
    logger.info(f"Anomalías reales: {true_anomalies}, Anomalías detectadas: {detected_anomalies}")
    
    # Calcular métricas de rendimiento
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(test_labels, predictions, labels=[-1, 1])
    report = classification_report(test_labels, predictions, labels=[-1, 1], target_names=["Anomalía", "Normal"])
    
    logger.info("\nMatriz de confusión:")
    logger.info(f"                  Predicho Anomalía    Predicho Normal")
    logger.info(f"Real Anomalía     {cm[0][0]}                {cm[0][1]}")
    logger.info(f"Real Normal       {cm[1][0]}                {cm[1][1]}")
    
    logger.info("\nInforme de clasificación:")
    logger.info(report)
    
    # Guardar resultados
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_results(results, os.path.join(results_dir, "batch_results.json"))
    
    # Visualizar resultados
    try:
        # Gráfico de puntuaciones de anomalía
        plt.figure(figsize=(12, 6))
        scores = [r.anomaly_score for r in results]
        plt.plot(scores, 'b-', alpha=0.7)
        plt.axhline(y=results[0].threshold, color='r', linestyle='--', label="Umbral")
        
        # Destacar anomalías reales
        anomaly_indices = np.where(test_labels == -1)[0]
        plt.scatter(anomaly_indices, [scores[i] for i in anomaly_indices], color='r', label="Anomalía Real")
        
        # Destacar anomalías detectadas
        detected_indices = [i for i, r in enumerate(results) if r.is_anomaly]
        plt.scatter(detected_indices, [scores[i] for i in detected_indices], 
                   marker='o', edgecolor='g', facecolor='none', s=80, label="Anomalía Detectada")
        
        plt.title("Puntuaciones de Anomalía")
        plt.xlabel("Índice de Muestra")
        plt.ylabel("Puntuación")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(results_dir, "batch_anomaly_scores.png"))
        logger.info(f"Visualización guardada en {os.path.join(results_dir, 'batch_anomaly_scores.png')}")
        
        # Si hay anomalías detectadas, mostrar contribución de características
        anomalies = [r for r in results if r.is_anomaly]
        if anomalies:
            fig = detector.plot_feature_contributions(anomalies[0])
            if fig:
                fig.savefig(os.path.join(results_dir, "feature_contributions.png"))
                logger.info(f"Visualización de contribuciones guardada en {os.path.join(results_dir, 'feature_contributions.png')}")
    
    except Exception as e:
        logger.error(f"Error al visualizar resultados: {e}")


def demo_time_series() -> None:
    """
    Demostración de detección de anomalías en series temporales.
    """
    logger.info("Iniciando demostración de series temporales")
    
    # Generar datos de serie temporal
    data, labels, feature_names = generate_time_series_data(
        n_points=1500,
        n_features=5,
        n_anomalies=75,
        seasonal_patterns=True
    )
    
    # Dividir en entrenamiento (solo normales) y test
    # Para series temporales, usaremos los primeros puntos para entrenamiento
    n_train = 500
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    
    # Eliminar anomalías del conjunto de entrenamiento
    train_normal_indices = train_labels == 1
    train_data = train_data[train_normal_indices]
    
    test_data = data[n_train:]
    test_labels = labels[n_train:]
    
    # Inicializar detector
    logger.info("Inicializando detector de anomalías para series temporales")
    detector = AnomalyDetector()
    
    # Entrenar detector
    logger.info("Entrenando detector con datos normales")
    detector.fit(train_data, feature_names=feature_names)
    
    # Simular procesamiento en tiempo real
    logger.info("Simulando procesamiento en tiempo real")
    window_size = 100
    results = []
    
    for i in range(0, len(test_data), window_size // 2):
        end_idx = min(i + window_size, len(test_data))
        batch = test_data[i:end_idx]
        batch_results = detector.process_batch(batch)
        results.extend(batch_results)
        
        # Actualizar modelo cada 5 ventanas con datos no anómalos
        if i > 0 and i % (window_size * 5) == 0:
            # Actualización incremental: mezclar datos normales recientes y antiguos
            recent_normal = [r.preprocessed_data for r in results[-100:] if not r.is_anomaly]
            if recent_normal:
                recent_normal = np.vstack(recent_normal)
                logger.info(f"Actualizando modelo incrementalmente con {len(recent_normal)} puntos normales recientes")
                detector.update_model(recent_normal, incremental=True)
    
    # Analizar resultados
    predictions = [1 if not r.is_anomaly else -1 for r in results]
    predictions = predictions[:len(test_labels)]  # Asegurar misma longitud
    
    # Calcular métricas de rendimiento
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(test_labels, predictions, labels=[-1, 1])
    report = classification_report(test_labels, predictions, labels=[-1, 1], target_names=["Anomalía", "Normal"])
    
    logger.info("\nRendimiento en la detección de series temporales:")
    logger.info(f"Anomalías reales: {np.sum(test_labels == -1)}, Anomalías detectadas: {np.sum(np.array(predictions) == -1)}")
    logger.info("\nMatriz de confusión:")
    logger.info(f"                  Predicho Anomalía    Predicho Normal")
    logger.info(f"Real Anomalía     {cm[0][0]}                {cm[0][1]}")
    logger.info(f"Real Normal       {cm[1][0]}                {cm[1][1]}")
    logger.info("\nInforme de clasificación:")
    logger.info(report)
    
    # Guardar resultados
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_results(results[:len(test_labels)], os.path.join(results_dir, "time_series_results.json"))
    
    # Visualizar resultados
    try:
        # Gráfico de serie temporal con anomalías
        plt.figure(figsize=(15, 8))
        
        # Mostrar serie temporal (primera característica)
        feature_idx = 0
        plt.plot(test_data[:, feature_idx], 'b-', alpha=0.7, label=f"{feature_names[feature_idx]}")
        
        # Destacar anomalías reales
        anomaly_indices = np.where(test_labels == -1)[0]
        plt.scatter(anomaly_indices, test_data[anomaly_indices, feature_idx], 
                   color='r', label="Anomalía Real")
        
        # Destacar anomalías detectadas
        detected_indices = [i for i, p in enumerate(predictions) if p == -1]
        plt.scatter(detected_indices, test_data[detected_indices, feature_idx], 
                   marker='o', edgecolor='g', facecolor='none', s=80, label="Anomalía Detectada")
        
        plt.title("Detección de Anomalías en Serie Temporal")
        plt.xlabel("Tiempo")
        plt.ylabel(feature_names[feature_idx])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(results_dir, "time_series_anomalies.png"))
        logger.info(f"Visualización guardada en {os.path.join(results_dir, 'time_series_anomalies.png')}")
        
        # Gráfico de puntuaciones de anomalía
        plt.figure(figsize=(15, 8))
        scores = [r.anomaly_score for r in results]
        thresholds = [r.threshold for r in results]
        
        plt.plot(scores[:len(test_labels)], 'b-', alpha=0.7, label="Puntuación de Anomalía")
        plt.plot(thresholds[:len(test_labels)], 'r--', alpha=0.7, label="Umbral Adaptativo")
        
        plt.title("Puntuaciones de Anomalía y Umbral Adaptativo")
        plt.xlabel("Tiempo")
        plt.ylabel("Puntuación")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(results_dir, "time_series_scores.png"))
        logger.info(f"Visualización guardada en {os.path.join(results_dir, 'time_series_scores.png')}")
    
    except Exception as e:
        logger.error(f"Error al visualizar resultados: {e}")


def demo_regimes() -> None:
    """
    Demostración de detección de anomalías con cambios de régimen.
    """
    logger.info("Iniciando demostración de detección de regímenes")
    
    # Generar datos con cambios de régimen explícitos
    # Para este ejemplo, generaremos tres regímenes distintos
    n_points_per_regime = 300
    n_features = 5
    n_anomalies_per_regime = 15
    
    np.random.seed(42)
    
    # Régimen 1: bajo nivel, poco ruido
    regime1_mean = np.array([0, 0, 0, 0, 0])
    regime1_std = 0.5
    regime1_data = np.random.normal(regime1_mean, regime1_std, (n_points_per_regime, n_features))
    regime1_labels = np.ones(n_points_per_regime)
    
    # Régimen 2: nivel medio, ruido moderado
    regime2_mean = np.array([2, 2, 2, 2, 2])
    regime2_std = 1.0
    regime2_data = np.random.normal(regime2_mean, regime2_std, (n_points_per_regime, n_features))
    regime2_labels = np.ones(n_points_per_regime)
    
    # Régimen 3: alto nivel, mucho ruido
    regime3_mean = np.array([5, 5, 5, 5, 5])
    regime3_std = 2.0
    regime3_data = np.random.normal(regime3_mean, regime3_std, (n_points_per_regime, n_features))
    regime3_labels = np.ones(n_points_per_regime)
    
    # Añadir anomalías a cada régimen
    # Las anomalías son diferentes según el régimen
    
    # Régimen 1: anomalías por valores altos
    anomaly_indices = np.random.choice(n_points_per_regime, n_anomalies_per_regime, replace=False)
    for idx in anomaly_indices:
        regime1_data[idx] = regime1_mean + 5 * regime1_std
        regime1_labels[idx] = -1
    
    # Régimen 2: anomalías por valores bajos o muy altos
    anomaly_indices = np.random.choice(n_points_per_regime, n_anomalies_per_regime, replace=False)
    for idx in anomaly_indices:
        anomaly_type = np.random.randint(0, 2)
        if anomaly_type == 0:
            regime2_data[idx] = regime2_mean - 3 * regime2_std
        else:
            regime2_data[idx] = regime2_mean + 6 * regime2_std
        regime2_labels[idx] = -1
        
    # Régimen 3: anomalías por correlaciones inusuales
    anomaly_indices = np.random.choice(n_points_per_regime, n_anomalies_per_regime, replace=False)
    for idx in anomaly_indices:
        # Hacer que todas las características sean iguales
        uniform_value = np.random.normal(regime3_mean[0], regime3_std)
        regime3_data[idx] = np.array([uniform_value] * n_features)
        regime3_labels[idx] = -1
    
    # Concatenar todos los regímenes
    all_data = np.vstack([regime1_data, regime2_data, regime3_data])
    all_labels = np.hstack([regime1_labels, regime2_labels, regime3_labels])
    
    # Definir nombres de características
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # También crear etiquetas de régimen para evaluación
    regime_labels = np.array(['low'] * n_points_per_regime + ['medium'] * n_points_per_regime + ['high'] * n_points_per_regime)
    
    logger.info(f"Datos generados con 3 regímenes: {len(all_data)} puntos, {np.sum(all_labels == -1)} anomalías")
    
    # Dividir en entrenamiento (una parte de cada régimen) y prueba
    train_ratio = 0.5
    n_train_per_regime = int(n_points_per_regime * train_ratio)
    
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    test_regime = []
    
    for i, start_idx in enumerate([0, n_points_per_regime, 2 * n_points_per_regime]):
        # Seleccionar índices para este régimen
        normal_indices = np.where(all_labels[start_idx:start_idx + n_points_per_regime] == 1)[0] + start_idx
        anomaly_indices = np.where(all_labels[start_idx:start_idx + n_points_per_regime] == -1)[0] + start_idx
        
        # Dividir normales en entrenamiento y prueba
        np.random.shuffle(normal_indices)
        train_indices = normal_indices[:n_train_per_regime]
        test_indices = np.concatenate([normal_indices[n_train_per_regime:], anomaly_indices])
        
        # Añadir a los conjuntos de train y test
        train_data.append(all_data[train_indices])
        train_labels.append(all_labels[train_indices])
        test_data.append(all_data[test_indices])
        test_labels.append(all_labels[test_indices])
        test_regime.append(regime_labels[test_indices])
    
    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)
    test_data = np.vstack(test_data)
    test_labels = np.hstack(test_labels)
    test_regime = np.hstack(test_regime)
    
    # Inicializar detector
    logger.info("Inicializando detector de anomalías con detección de regímenes")
    detector = AnomalyDetector()
    
    # Entrenar detector
    logger.info("Entrenando detector con datos de todos los regímenes")
    detector.fit(train_data, feature_names=feature_names)
    
    # Procesar datos de prueba
    logger.info(f"Procesando {len(test_data)} puntos de prueba")
    results = detector.process_batch(test_data, explain_anomalies=True)
    
    # Extraer regímenes detectados para cada punto
    detected_regimes = [result.regime for result in results]
    
    # Calcular métricas de rendimiento
    predictions = [1 if not r.is_anomaly else -1 for r in results]
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(test_labels, predictions, labels=[-1, 1])
    report = classification_report(test_labels, predictions, labels=[-1, 1], target_names=["Anomalía", "Normal"])
    
    logger.info("\nRendimiento en la detección con múltiples regímenes:")
    logger.info(f"Anomalías reales: {np.sum(test_labels == -1)}, Anomalías detectadas: {np.sum(np.array(predictions) == -1)}")
    logger.info("\nMatriz de confusión:")
    logger.info(f"                  Predicho Anomalía    Predicho Normal")
    logger.info(f"Real Anomalía     {cm[0][0]}                {cm[0][1]}")
    logger.info(f"Real Normal       {cm[1][0]}                {cm[1][1]}")
    logger.info("\nInforme de clasificación:")
    logger.info(report)
    
    # Evaluar precisión por régimen
    regime_accuracy = {}
    for regime in set(test_regime):
        regime_indices = [i for i, r in enumerate(test_regime) if r == regime]
        regime_true = test_labels[regime_indices]
        regime_pred = np.array(predictions)[regime_indices]
        
        correct = np.sum(regime_true == regime_pred)
        total = len(regime_indices)
        accuracy = correct / total if total > 0 else 0
        
        regime_accuracy[regime] = accuracy
        
    logger.info("\nPrecisión por régimen:")
    for regime, accuracy in regime_accuracy.items():
        logger.info(f"  {regime}: {accuracy:.4f}")
    
    # Guardar resultados
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_results(results, os.path.join(results_dir, "regime_results.json"))
    
    # Visualizar resultados
    try:
        # Gráfico de puntuaciones por régimen
        plt.figure(figsize=(15, 10))
        
        regimes = list(set(detected_regimes))
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
        regime_colors = {regime: colors[i % len(colors)] for i, regime in enumerate(regimes)}
        
        # Graficar puntuaciones por régimen
        for regime in regimes:
            regime_indices = [i for i, r in enumerate(detected_regimes) if r == regime]
            
            if not regime_indices:
                continue
                
            scores = [results[i].anomaly_score for i in regime_indices]
            thresholds = [results[i].threshold for i in regime_indices]
            x_values = np.array(regime_indices)
            
            plt.scatter(x_values, scores, c=regime_colors[regime], label=f"Régimen: {regime}", alpha=0.7)
            plt.plot(x_values, thresholds, c=regime_colors[regime], linestyle='--', alpha=0.5)
        
        # Marcar anomalías reales
        anomaly_indices = np.where(test_labels == -1)[0]
        plt.scatter(anomaly_indices, [1.1] * len(anomaly_indices), marker='x', color='red', s=100, label='Anomalía Real')
        
        # Marcar anomalías detectadas
        detected_indices = [i for i, r in enumerate(results) if r.is_anomaly]
        plt.scatter(detected_indices, [1.05] * len(detected_indices), marker='o', facecolors='none', 
                   edgecolors='green', s=150, label='Anomalía Detectada')
        
        plt.title("Detección de Anomalías en Múltiples Regímenes")
        plt.xlabel("Índice de Muestra")
        plt.ylabel("Puntuación de Anomalía")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(results_dir, "regime_detection.png"))
        logger.info(f"Visualización guardada en {os.path.join(results_dir, 'regime_detection.png')}")
        
    except Exception as e:
        logger.error(f"Error al visualizar resultados: {e}")


def demo_interactive() -> None:
    """
    Demostración interactiva donde el usuario puede probar puntos específicos.
    """
    logger.info("Iniciando demo interactiva")
    
    # Generar datos sintéticos para entrenamiento
    normal_data, _, feature_names = generate_synthetic_data(
        n_normal=800,
        n_anomalies=0,
        n_features=5
    )
    
    # Inicializar detector
    detector = AnomalyDetector()
    detector.fit(normal_data, feature_names=feature_names)
    
    print("\n===== Demo Interactiva de Detección de Anomalías =====")
    print(f"El detector ha sido entrenado con {len(normal_data)} puntos normales.")
    print(f"Características: {', '.join(feature_names)}")
    print("\nInstructions:")
    print("- Ingrese valores para cada característica separados por espacios")
    print("- Ingrese 'q' o 'exit' para salir")
    print("- Ingrese 'random' para generar un punto aleatorio normal")
    print("- Ingrese 'anomaly' para generar un punto anómalo aleatorio")
    
    while True:
        print("\n" + "-" * 50)
        user_input = input("Ingrese valores o comando: ").strip().lower()
        
        if user_input in ['q', 'exit']:
            break
        
        try:
            if user_input == 'random':
                # Generar punto normal aleatorio
                point = np.random.normal(0, 1, len(feature_names))
                print(f"Punto generado: {point}")
            
            elif user_input == 'anomaly':
                # Generar punto anómalo aleatorio
                point = np.random.normal(5, 1, len(feature_names))
                print(f"Punto anómalo generado: {point}")
            
            else:
                # Parsear valores ingresados por el usuario
                values = [float(x) for x in user_input.split()]
                if len(values) != len(feature_names):
                    print(f"Error: Se requieren {len(feature_names)} valores.")
                    continue
                point = np.array(values)
            
            # Detectar anomalías
            result = detector.detect(point, explain=True)
            
            # Mostrar resultado
            print("\nResultado de detección:")
            print(f"{'ANOMALÍA DETECTADA!' if result.is_anomaly else 'Normal'}")
            print(f"Puntuación: {result.anomaly_score:.4f} (Umbral: {result.threshold:.4f})")
            print(f"Régimen detectado: {result.regime}")
            
            if result.is_anomaly and result.feature_contributions:
                print("\nContribuciones de características:")
                for i, (feature, score) in enumerate(result.feature_contributions[:3]):
                    print(f"  {i+1}. {feature}: {score:.4f}")
            
            if result.explanation:
                print(f"\nExplicación: {result.explanation}")
                
        except Exception as e:
            print(f"Error: {str(e)}")


def demo_real_time_simulation() -> None:
    """
    Simula procesamiento en tiempo real con visualización dinámica.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        logger.info("Iniciando simulación en tiempo real")
        
        # Generar datos base
        ts_data, ts_labels, feature_names = generate_time_series_data(
            n_points=2000,
            n_features=3,
            n_anomalies=20,
            seasonal_patterns=True
        )
        
        # Dividir en entrenamiento y simulación
        train_size = 500
        train_data = ts_data[:train_size]
        
        # Inicializar detector
        detector = AnomalyDetector()
        detector.fit(train_data, feature_names=feature_names)
        
        # Configuración de visualización
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Simulación de Detección de Anomalías en Tiempo Real", fontsize=16)
        
        # Clase para mantener el estado de la animación
        class AnimationState:
            def __init__(self):
                self.window_size = 100
                self.visible_data = []
                self.visible_scores = []
                self.visible_thresholds = []
                self.visible_anomalies = []
                self.sim_index = train_size
                self.anomalies_detected = 0
                self.points_processed = 0
                self.current_regime = "normal"
        
        # Crear estado de animación
        state = AnimationState()
        
        # Líneas para visualización
        line1, = ax1.plot([], [], 'c-', label="Valor")
        anomaly_points = ax1.scatter([], [], color='red', s=100, label="Anomalía")
        
        line2, = ax2.plot([], [], 'b-', label="Puntuación")
        threshold_line, = ax2.plot([], [], 'r--', label="Umbral")
        anomaly_scores = ax2.scatter([], [], color='red', s=50)
        
        # Configurar ejes
        ax1.set_xlim(0, state.window_size)
        ax1.set_ylim(-3, 3)
        ax1.set_title("Datos de Serie Temporal")
        ax1.set_ylabel("Valor")
        ax1.legend(loc="upper right")
        ax1.grid(True)
        
        ax2.set_xlim(0, state.window_size)
        ax2.set_ylim(-0.1, 1.5)
        ax2.set_title("Puntuación de Anomalía")
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("Puntuación")
        ax2.legend(loc="upper right")
        ax2.grid(True)
        
        # Texto para estadísticas
        stats_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, 
                             color='white', fontsize=10, verticalalignment='top')
        
        # Función para actualizar la visualización
        def update(frame):
            # Check if we've reached the end of data
            if state.sim_index >= len(ts_data):
                stats_text.set_text("Simulación completada")
                return line1, anomaly_points, line2, threshold_line, anomaly_scores, stats_text
            
            # Procesar nuevo punto
            current_point = ts_data[state.sim_index]
            result = detector.detect(current_point, explain=False)
            state.points_processed += 1
            state.current_regime = result.regime
            
            # Actualizar historial
            state.visible_data.append(current_point[0])  # Primera característica
            state.visible_scores.append(result.anomaly_score)
            state.visible_thresholds.append(result.threshold)
            
            # Añadir a las anomalías si es necesario
            if result.is_anomaly:
                state.anomalies_detected += 1
                state.visible_anomalies.append(len(state.visible_data) - 1)
            
            # Limitar tamaño de ventana
            if len(state.visible_data) > state.window_size:
                # Actualizar índices de anomalías
                state.visible_anomalies = [idx - 1 for idx in state.visible_anomalies if idx > 0]
                # Eliminar punto más antiguo
                state.visible_data.pop(0)
                state.visible_scores.pop(0)
                state.visible_thresholds.pop(0)
            
            # Actualizar visualización
            x_indices = list(range(len(state.visible_data)))
            
            line1.set_data(x_indices, state.visible_data)
            line2.set_data(x_indices, state.visible_scores)
            threshold_line.set_data(x_indices, state.visible_thresholds)
            
            # Actualizar puntos anómalos
            if state.visible_anomalies:
                anomaly_points.set_offsets([[idx, state.visible_data[idx]] for idx in state.visible_anomalies])
                anomaly_scores.set_offsets([[idx, state.visible_scores[idx]] for idx in state.visible_anomalies])
            else:
                anomaly_points.set_offsets(np.empty((0, 2)))
                anomaly_scores.set_offsets(np.empty((0, 2)))
            
            # Ajustar límites si es necesario
            if state.visible_data:
                min_y = min(state.visible_data) - 0.5
                max_y = max(state.visible_data) + 0.5
                ax1.set_ylim(min_y, max_y)
                
                max_score = max(state.visible_scores) + 0.2
                ax2.set_ylim(-0.1, max(1.5, max_score))
            
            # Actualizar estadísticas y título
            stats_text.set_text(f"Procesados: {state.points_processed}   Anomalías: {state.anomalies_detected}")
            ax1.set_title(f"Datos de Serie Temporal - Régimen: {state.current_regime}")
            
            # Incrementar contador
            state.sim_index += 1
            
            return line1, anomaly_points, line2, threshold_line, anomaly_scores, stats_text
        
        # Inicializar valores para primera llamada de la animación
        update(0)  # Call once to set up initial state
        
        # Crear animación
        ani = FuncAnimation(fig, update, frames=None, interval=100, blit=True)
        
        # Añadir botón de parada
        from matplotlib.widgets import Button
        stop_ax = plt.axes([0.85, 0.01, 0.1, 0.04])
        stop_button = Button(stop_ax, 'Detener', color='lightgray', hovercolor='red')
        
        def stop_animation(event):
            ani.event_source.stop()
            plt.text(0.5, 0.5, "Simulación Detenida", 
                    transform=fig.transFigure, ha='center', fontsize=20)
            plt.draw()
        
        stop_button.on_clicked(stop_animation)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        # Mostrar visualización interactiva
        print("\n===== Simulación en Tiempo Real =====")
        print("Presione el botón 'Detener' o cierre la ventana para terminar")
        plt.show()
        
    except ImportError as e:
        logger.error(f"Error al importar bibliotecas necesarias: {e}")
        print("Esta demo requiere matplotlib con soporte de animación interactiva.")
    except Exception as e:
        logger.error(f"Error en simulación: {e}")
        import traceback
        traceback.print_exc()


def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sistema de Detección de Anomalías para Proyecciones Multidimensionales en Tiempo Real"
    )
    
    parser.add_argument(
        "--demo", 
        choices=["batch", "time_series", "regimes", "interactive", "simulation"],
        default="batch",
        help="Tipo de demostración a ejecutar"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="./results",
        help="Directorio de salida para resultados y visualizaciones"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Ruta a archivo de configuración JSON opcional"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Mostrar información detallada durante la ejecución"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parsear argumentos de línea de comandos
    args = parse_arguments()
    
    # Configurar nivel de logging según verbosidad
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Cargar configuración si se especifica
    if args.config:
        if os.path.exists(args.config):
            from config import load_config_from_json
            config = load_config_from_json(args.config)
            logger.info(f"Configuración cargada desde {args.config}")
        else:
            logger.warning(f"Archivo de configuración no encontrado: {args.config}")
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output, exist_ok=True)
    
    # Ejecutar demo seleccionada
    try:
        print("=" * 70)
        print(f"  Sistema de Detección de Anomalías - Modo: {args.demo}")
        print("=" * 70)
        
        if args.demo:
            logger.info(f"Ejecutando demostración: {args.demo}")

        if args.demo == "batch":
            demo_batch_processing()
        elif args.demo == "time_series":
            demo_time_series()
        elif args.demo == "regimes":
            demo_regimes()
        elif args.demo == "interactive":
            demo_interactive()
        elif args.demo == "simulation":
            demo_real_time_simulation()
        else:
            logger.error(f"Modo de demostración no reconocido: {args.demo}")
        
        print("\n" + "=" * 70)
        print("  Demostración completada con éxito")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nEjecución interrumpida por el usuario.")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
