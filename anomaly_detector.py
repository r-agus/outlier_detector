#!/usr/bin/env python3
"""
Módulo principal para la detección de anomalías.

Este módulo integra todos los componentes del sistema (preprocesamiento,
modelos, umbrales adaptativos, detección de regímenes, etc.) y proporciona
una interfaz unificada para la detección de anomalías en tiempo real.
"""

from math import log
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# Importar componentes del sistema
from config import config
from preprocesing import DataPreprocessor, StreamPreprocessor
from models import ModelManager, BaseAnomalyDetector, EnsembleDetector
from thresholds import ThresholdManager, MovingStatsThreshold, ProbabilisticThreshold, ContextualThreshold
from regime_detector import StatisticalRegimeDetector, TimeBasedRegimeDetector, HybridRegimeDetector
from feature_contribution import ContributionVisualizer, EnsembleFeatureAnalyzer

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('anomaly_detector')

class AnomalyDetectionResult:
    """
    Clase para almacenar y gestionar resultados de detección de anomalías.
    """
    
    def __init__(self, 
                timestamp: float,
                data_point: np.ndarray,
                preprocessed_data: np.ndarray,
                anomaly_score: float,
                is_anomaly: bool,
                threshold: float,
                detector_name: str,
                regime: str = "normal"):
        """
        Inicializa un resultado de detección de anomalías.
        
        Args:
            timestamp: Timestamp de la detección
            data_point: Datos originales
            preprocessed_data: Datos preprocesados
            anomaly_score: Puntuación de anomalía
            is_anomaly: Si se considera anomalía
            threshold: Umbral utilizado
            detector_name: Nombre del detector que identificó la anomalía
            regime: Régimen operacional actual
        """
        self.timestamp = timestamp
        self.data_point = data_point
        self.preprocessed_data = preprocessed_data
        self.anomaly_score = anomaly_score
        self.is_anomaly = is_anomaly
        self.threshold = threshold
        self.detector_name = detector_name
        self.regime = regime
        self.feature_contributions = None
        self.explanation = None
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el resultado a un diccionario.
        
        Returns:
            Diccionario con los resultados
        """
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "detector": self.detector_name,
            "regime": self.regime,
            "features": self.feature_contributions
        }
        
    def add_feature_contributions(self, contributions: List[Tuple[str, float]]) -> None:
        """
        Añade información sobre la contribución de las características.
        
        Args:
            contributions: Lista de tuplas (nombre_característica, puntuación)
        """
        self.feature_contributions = contributions
        
    def add_explanation(self, explanation: str) -> None:
        """
        Añade una explicación en lenguaje natural de la anomalía.
        
        Args:
            explanation: Texto explicativo
        """
        self.explanation = explanation
        
    def __str__(self) -> str:
        """Representación en texto del resultado."""
        status = "ANOMALÍA" if self.is_anomaly else "Normal"
        time_str = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        result = f"[{time_str}] {status} - Score: {self.anomaly_score:.4f} (Umbral: {self.threshold:.4f}, Detector: {self.detector_name}, Régimen: {self.regime})"
        
        if self.is_anomaly and self.feature_contributions:
            result += "\nCaracterísticas contribuyentes:"
            for i, (feature, score) in enumerate(self.feature_contributions[:3]):
                result += f"\n  {i+1}. {feature}: {score:.4f}"
                
        if self.explanation:
            result += f"\nExplicación: {self.explanation}"
            
        return result


class AnomalyDetector:
    """
    Clase principal que integra todos los componentes del sistema para la detección de anomalías.
    """
    
    def __init__(self, config_override: Dict = None):
        """
        Inicializa el detector de anomalías con todos sus componentes.
        
        Args:
            config_override: Configuración opcional para sobrescribir la configuración por defecto
        """
        # Aplicar configuración personalizada si se proporciona
        if config_override:
            config.update_from_dict(config_override)
            
        logger.info("Inicializando sistema de detección de anomalías...")
        
        # Inicializar componentes
        self._init_preprocessor()
        self._init_models()
        self._init_thresholds()
        self._init_regime_detector()
        
        # Estado interno
        self.is_trained = False
        self.current_regime = "normal"
        self.anomaly_history = deque(maxlen=1000)  # Historial de anomalías recientes
        self.data_history = deque(maxlen=config.preprocessing.window_size)  # Historial de datos recientes
        self.feature_names = None
        
        # Callbacks para notificación de anomalías
        self.anomaly_callbacks = []
        
        # Métricas de rendimiento
        self.performance_metrics = {
            "total_points_processed": 0,
            "anomalies_detected": 0,
            "processing_times": deque(maxlen=100),  # Tiempos de procesamiento recientes
            "last_evaluation_time": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
        
        logger.info("Sistema de detección de anomalías inicializado.")
    
    def _init_preprocessor(self) -> None:
        """Inicializa el preprocesador de datos."""
        logger.info("Inicializando preprocesador de datos...")
        
        # Para procesamiento en tiempo real, usar StreamPreprocessor
        self.preprocessor = StreamPreprocessor()
        
        # Para procesamiento por lotes, usar DataPreprocessor estándar
        self.batch_preprocessor = DataPreprocessor()
    
    def _init_models(self) -> None:
        """Inicializa los modelos de detección de anomalías."""
        logger.info("Inicializando modelos de detección...")
        
        # Usar ModelManager para gestionar modelos
        self.model_manager = ModelManager()
        
        # Crear modelos básicos predefinidos
        self.lof = self.model_manager.create_detector("lof", "LOF")
        self.model_manager.add_detector(self.lof)  # Add to active detectors
        
        self.iforest = self.model_manager.create_detector("isolation_forest", "IForest")
        self.model_manager.add_detector(self.iforest)  # Add to active detectors
        
        self.ocsvm = self.model_manager.create_detector("one_class_svm", "OCSVM")
        self.model_manager.add_detector(self.ocsvm)  # Add to active detectors
        
        self.autoencoder = self.model_manager.create_detector("autoencoder", "Autoencoder")
        self.model_manager.add_detector(self.autoencoder)  # Add to active detectors
        
        # Crear un detector de ensamble por defecto
        self.ensemble = self.model_manager.create_ensemble(
            detector_names=["LOF", "IForest", "OCSVM"],
            method="voting",
            name="Ensemble"
        )
        
        # Seleccionar detector activo por defecto
        self.active_detector = self.ensemble
        
        # Configurar detector para cada capa de procesamiento
        self.stratified_detectors = {
            "fast_layer": self.iforest,     # Rápido para primera línea
            "medium_layer": self.lof,       # Precisión media para segunda línea
            "complex_layer": self.ensemble  # Alta precisión para análisis profundo
        }
    
    def _init_thresholds(self) -> None:
        """Inicializa los mecanismos de umbrales adaptativos."""
        logger.info("Inicializando umbrales adaptativos...")
        
        # Crear gestor de umbrales
        self.threshold_manager = ThresholdManager()
        
        # Crear diferentes estrategias de umbral
        self.moving_stats_threshold = MovingStatsThreshold(name="moving_stats")
        self.probabilistic_threshold = ProbabilisticThreshold(name="probabilistic")
        self.contextual_threshold = ContextualThreshold(name="contextual")
        
        # Añadir umbrales al gestor
        self.threshold_manager.add_threshold(self.moving_stats_threshold)
        self.threshold_manager.add_threshold(self.probabilistic_threshold)
        self.threshold_manager.add_threshold(self.contextual_threshold)
        
        # Configurar meta-umbral
        self.threshold_manager.setup_meta_threshold(
            threshold_names=["moving_stats", "probabilistic", "contextual"],
            weights=[1.0, 1.5, 1.2]
        )
        
        # Establecer estrategia de umbral por defecto
        self.threshold_manager.set_current_strategy("meta_threshold")
    
    def _init_regime_detector(self) -> None:
        """Inicializa el detector de regímenes operacionales."""
        logger.info("Inicializando detector de regímenes...")
        
        # Crear detectores de régimen
        self.statistical_regime_detector = StatisticalRegimeDetector("statistical_regime")
        self.time_regime_detector = TimeBasedRegimeDetector("time_regime")
        
        # Configurar detector híbrido que combine ambos enfoques
        self.regime_detector = HybridRegimeDetector("hybrid_regime")
        
        # Añadir callback para cambios de régimen
        self.regime_detector.add_callback(self._on_regime_change)
    
    def _on_regime_change(self, new_regime: str, old_regime: str) -> None:
        """
        Callback que se ejecuta cuando cambia el régimen operacional.
        
        Args:
            new_regime: Nuevo régimen detectado
            old_regime: Régimen anterior
        """
        logger.info(f"Cambio de régimen detectado: {old_regime} -> {new_regime}")
        
        # Actualizar régimen actual
        self.current_regime = new_regime
        
        # Notificar al umbral contextual
        if hasattr(self, 'contextual_threshold'):
            self.contextual_threshold.set_regime(new_regime)
            
        # Adaptar estrategia de detección según el régimen
        self._adapt_detection_strategy(new_regime)
    
    def _adapt_detection_strategy(self, regime: str) -> None:
        """
        Adapta la estrategia de detección al régimen actual.
        
        Args:
            regime: Régimen actual
        """
        # Adaptar según el régimen
        if regime == "high_activity":
            # En alta actividad, priorizar precisión sobre recall
            self.threshold_manager.set_current_strategy("probabilistic")  # Más conservador
            logger.info("Adaptando a régimen de alta actividad: umbral probabilístico")
            
        elif regime == "low_activity":
            # En baja actividad, usar enfoque más sensible
            self.threshold_manager.set_current_strategy("moving_stats")  # Más sensible
            logger.info("Adaptando a régimen de baja actividad: umbral de estadísticas móviles")
            
        else:  # normal o cualquier otro
            # Para régimen normal, usar el meta-umbral balanceado
            self.threshold_manager.set_current_strategy("meta_threshold")
            logger.info("Adaptando a régimen normal: meta-umbral")
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Establece los nombres de las características para reportes y visualizaciones.
        
        Args:
            feature_names: Lista con los nombres de las características
        """
        self.feature_names = feature_names
        logger.info(f"Nombres de características establecidos: {feature_names}")
    
    def add_anomaly_callback(self, callback: Callable[[AnomalyDetectionResult], None]) -> None:
        """
        Añade un callback que se ejecutará cuando se detecte una anomalía.
        
        Args:
            callback: Función que recibe un objeto AnomalyDetectionResult
        """
        self.anomaly_callbacks.append(callback)
        logger.info("Callback de detección de anomalías añadido")
    
    def fit(self, training_data: np.ndarray, feature_names: List[str] = None) -> None:
        """
        Entrena todos los componentes del sistema con datos normales.
        
        Args:
            training_data: Datos normales para entrenamiento
            feature_names: Nombres de las características (opcional)
        """
        start_time = time.time()
        logger.info(f"Iniciando entrenamiento con {len(training_data)} puntos de datos...")
        
        # Establecer nombres de características si se proporcionan
        if feature_names:
            self.set_feature_names(feature_names)
        
        # Preprocesar datos de entrenamiento
        self.batch_preprocessor.fit(training_data)
        preprocessed_data = self.batch_preprocessor.normalize(training_data)
        
        # Entrenar modelos individuales
        for name, detector in self.model_manager.active_detectors.items():
            logger.info(f"Entrenando modelo: {name}")
            detector.fit(preprocessed_data)
        
        # Entrenar explícitamente el modelo de ensamble después de los detectores individuales
        logger.info(f"Entrenando modelo de ensamble: {self.ensemble.name}")
        self.ensemble.fit(preprocessed_data)
        
        # Calcular puntuaciones iniciales para establecer umbrales
        scores_dict = {}
        for name, detector in self.model_manager.active_detectors.items():
            scores = detector.decision_function(preprocessed_data)
            scores_dict[name] = scores
            
            # Actualizar umbral del detector
            detector.update_threshold(scores, percentile=95)
        
        # Ahora el ensamble está entrenado, podemos calcular sus puntuaciones
        ensemble_scores = self.ensemble.decision_function(preprocessed_data)
        self.ensemble.update_threshold(ensemble_scores, percentile=95)
        
        # Inicializar umbrales adaptativos
        self.moving_stats_threshold.update(ensemble_scores)
        self.probabilistic_threshold.update(ensemble_scores)
        self.contextual_threshold.update(ensemble_scores)
        
        # Entrenar detector de regímenes si hay suficientes datos
        if len(training_data) > 50:
            self.regime_detector.fit(preprocessed_data)
        
        # Inicializar analizador de contribución
        self.contribution_analyzer = EnsembleFeatureAnalyzer(
            self.active_detector, 
            feature_names=self.feature_names
        )
        
        # Guardar una muestra de datos normales para comparaciones futuras
        self.normal_data_sample = preprocessed_data[:min(500, len(preprocessed_data))]
        
        self.is_trained = True
        
        # Actualizar métricas de rendimiento
        training_time = time.time() - start_time
        logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
    
    def detect(self, data_point: Union[np.ndarray, pd.Series, List], 
              explain: bool = False) -> AnomalyDetectionResult:
        """
        Detecta anomalías en un único punto de datos.
        
        Args:
            data_point: Punto de datos a analizar
            explain: Si generar explicación detallada en caso de anomalía
            
        Returns:
            Resultado del análisis de anomalías
        """
        start_time = time.time()
        
        # Verificar inicialización
        if not self.is_trained:
            raise RuntimeError("El detector no ha sido entrenado. Ejecute fit() primero.")
        
        # Convertir a numpy array si es necesario
        if isinstance(data_point, list):
            data_point = np.array(data_point)
        elif isinstance(data_point, pd.Series):
            data_point = data_point.values
        
        # Preprocesar el punto de datos
        preprocessed = self.preprocessor.process_stream(data_point)
        
        # Ensure preprocessed data has the right shape for decision_function
        # If it's a single sample, ensure it's 2D with shape (1, features)
        if preprocessed.ndim == 1:
            preprocessed_input = preprocessed.reshape(1, -1)
        else:
            preprocessed_input = preprocessed
        
        # Actualizar historial
        self.data_history.append(preprocessed)
        
        # Actualizar detector de régimen
        window_data = np.array(list(self.data_history))
        self.current_regime = self.regime_detector.update(window_data)
        
        # Calcular puntuación de anomalía con el detector activo
        anomaly_score = float(self.active_detector.decision_function(preprocessed_input)[0])
        
        # Actualizar umbral
        self.threshold_manager.update(np.array([anomaly_score]))
        current_threshold = self.threshold_manager.get_threshold()
        
        # Determinar si es anomalía
        is_anomaly = anomaly_score > current_threshold
        
        # Crear resultado
        result = AnomalyDetectionResult(
            timestamp=time.time(),
            data_point=data_point,
            preprocessed_data=preprocessed,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            threshold=current_threshold,
            detector_name=self.active_detector.name,
            regime=self.current_regime
        )
        
        # Si es anomalía, realizar análisis adicional
        if is_anomaly:
            # Incrementar contador
            self.performance_metrics["anomalies_detected"] += 1
            
            # Añadir al historial
            self.anomaly_history.append(result)
            
            # Generar explicación si se solicita
            if explain and hasattr(self, 'contribution_analyzer'):
                # Analizar contribución de características
                try:
                    contribution_result = self.contribution_analyzer.analyze(
                        preprocessed.reshape(1, -1), 
                        self.normal_data_sample
                    )
                    
                    # Añadir contribuciones al resultado
                    top_features = contribution_result.get("ranked_features", [])
                    result.add_feature_contributions(top_features)
                    
                    # Generar explicación en texto
                    explanation = self._generate_explanation(result, top_features)
                    result.add_explanation(explanation)
                    
                except Exception as e:
                    logger.error(f"Error al analizar contribución de características: {str(e)}")
            
            # Ejecutar callbacks de anomalía
            for callback in self.anomaly_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error en callback de anomalía: {str(e)}")
        
        # Actualizar métricas de rendimiento
        self.performance_metrics["total_points_processed"] += 1
        self.performance_metrics["processing_times"].append(time.time() - start_time)
        
        return result
    
    def process_batch(self, data_batch: Union[np.ndarray, pd.DataFrame], 
                     explain_anomalies: bool = False) -> List[AnomalyDetectionResult]:
        """
        Procesa un lote de datos para detección de anomalías.
        
        Args:
            data_batch: Lote de datos a procesar
            explain_anomalies: Si explicar anomalías detectadas
            
        Returns:
            Lista de resultados de detección
        """
        results = []
        
        logger.info(f"Iniciando procesamiento por lotes de {len(data_batch)} puntos de datos...")

        # Convertir a numpy si es DataFrame
        if isinstance(data_batch, pd.DataFrame):
            if self.feature_names is None and data_batch.columns is not None:
                self.feature_names = list(data_batch.columns)
            data_batch = data_batch.values
        
        # Procesar cada punto
        for i in range(data_batch.shape[0]):
            logger.info(f"Procesando punto {i+1}/{data_batch.shape[0]}...")
            result = self.detect(data_batch[i], explain=explain_anomalies)
            results.append(result)
            
        return results
    
    def _generate_explanation(self, result: AnomalyDetectionResult, 
                             top_features: List[Tuple[str, float]]) -> str:
        """
        Genera una explicación en lenguaje natural de la anomalía.
        
        Args:
            result: Resultado de detección
            top_features: Lista de características principales
            
        Returns:
            Texto explicativo
        """
        if not top_features:
            return "No se pudo generar una explicación detallada."
        
        # Extraer las 3 principales características
        top_three = [f"{name} ({score:.2f})" for name, score in top_features[:3]]
        
        # Generar explicación
        explanation = f"Anomalía detectada con puntuación {result.anomaly_score:.2f} "
        explanation += f"durante régimen '{result.regime}'. "
        explanation += f"Las características más contribuyentes son: {', '.join(top_three)}."
        
        return explanation
    
    def get_top_anomalies(self, top_n: int = 10) -> List[AnomalyDetectionResult]:
        """
        Obtiene las principales anomalías basadas en puntuación.
        
        Args:
            top_n: Número de anomalías a devolver
            
        Returns:
            Lista de resultados de detección
        """
        anomalies = list(self.anomaly_history)
        anomalies.sort(key=lambda x: x.anomaly_score, reverse=True)
        return anomalies[:min(top_n, len(anomalies))]
    
    def update_model(self, new_normal_data: np.ndarray, incremental: bool = True) -> None:
        """
        Actualiza los modelos con nuevos datos normales.
        
        Args:
            new_normal_data: Nuevos datos normales
            incremental: Si realizar actualización incremental
        """
        logger.info(f"Actualizando modelos con {len(new_normal_data)} nuevos puntos normales...")
        
        # Preprocesar datos
        preprocessed_data = self.batch_preprocessor.normalize(new_normal_data)
        
        # Actualizar modelos
        if incremental and hasattr(self.active_detector, 'partial_fit'):
            # Actualización incremental si está soportada
            self.active_detector.partial_fit(preprocessed_data)
            logger.info(f"Actualización incremental completada para {self.active_detector.name}")
        else:
            # Combinar con datos anteriores para reentrenamiento
            if hasattr(self, 'normal_data_sample'):
                combined_data = np.vstack([self.normal_data_sample, preprocessed_data])
                # Limitar tamaño si es necesario
                if len(combined_data) > 2000:  # Limitar a 2000 puntos para eficiencia
                    indices = np.random.choice(len(combined_data), 2000, replace=False)
                    combined_data = combined_data[indices]
            else:
                combined_data = preprocessed_data
                
            # Reentrenar modelos
            self.active_detector.fit(combined_data)
            logger.info(f"Reentrenamiento completo para {self.active_detector.name}")
            
            # Actualizar muestra de datos normales
            self.normal_data_sample = combined_data
        
        # Recalcular umbrales
        scores = self.active_detector.decision_function(preprocessed_data)
        self.threshold_manager.update(scores)
        
        logger.info("Actualización de modelos completada.")
    
    def register_feedback(self, result: AnomalyDetectionResult, was_anomaly: bool) -> None:
        """
        Registra retroalimentación sobre una detección para aprendizaje semi-supervisado.
        
        Args:
            result: Resultado de detección
            was_anomaly: Si realmente era una anomalía (validación)
        """
        # Registrar falsos positivos/negativos
        if result.is_anomaly and not was_anomaly:
            self.performance_metrics["false_positives"] += 1
            logger.info("Falso positivo registrado")
            
            # Ajustar umbral meta para reducir falsos positivos
            if hasattr(self.threshold_manager, 'thresholds') and 'meta_threshold' in self.threshold_manager.thresholds:
                meta = self.threshold_manager.thresholds['meta_threshold']
                meta.register_feedback(was_anomaly=False, predicted_anomaly=True)
            
        elif not result.is_anomaly and was_anomaly:
            self.performance_metrics["false_negatives"] += 1
            logger.info("Falso negativo registrado")
            
            # Ajustar umbral meta para reducir falsos negativos
            if hasattr(self.threshold_manager, 'thresholds') and 'meta_threshold' in self.threshold_manager.thresholds:
                meta = self.threshold_manager.thresholds['meta_threshold']
                meta.register_feedback(was_anomaly=True, predicted_anomaly=False)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del detector.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        metrics = self.performance_metrics.copy()
        
        # Calcular tiempo de procesamiento promedio
        if metrics["processing_times"]:
            metrics["avg_processing_time"] = sum(metrics["processing_times"]) / len(metrics["processing_times"])
        else:
            metrics["avg_processing_time"] = 0
            
        # Calcular tasa de anomalías
        if metrics["total_points_processed"] > 0:
            metrics["anomaly_rate"] = metrics["anomalies_detected"] / metrics["total_points_processed"]
        else:
            metrics["anomaly_rate"] = 0
            
        # Calcular precisión de la detección si hay feedback
        total_feedback = metrics["false_positives"] + metrics["false_negatives"]
        if total_feedback > 0:
            true_positives = metrics["anomalies_detected"] - metrics["false_positives"]
            metrics["precision"] = true_positives / (true_positives + metrics["false_positives"] + 1e-10)
        
        return metrics
    
    def plot_anomaly_history(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualiza el historial de puntuaciones de anomalía y umbrales.
        
        Args:
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import dates as mdates
            from datetime import datetime
            
            # Extraer datos históricos
            timestamps = []
            datetimes = []  # New list to store datetime objects
            scores = []
            thresholds = []
            anomaly_flags = []
            
            for result in self.anomaly_history:
                timestamps.append(result.timestamp)
                datetimes.append(datetime.fromtimestamp(result.timestamp))  # Convert to datetime
                scores.append(result.anomaly_score)
                thresholds.append(result.threshold)
                anomaly_flags.append(result.is_anomaly)
            
            # Convertir a arrays numpy
            if not timestamps:
                return None
                
            timestamps = np.array(timestamps)
            scores = np.array(scores)
            thresholds = np.array(thresholds)
            anomaly_flags = np.array(anomaly_flags)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=figsize)
            
            # Graficar puntuaciones y umbrales - using datetimes instead of raw timestamps
            ax.plot(datetimes, scores, 'b-', label="Puntuación de Anomalía")
            ax.plot(datetimes, thresholds, 'r--', label="Umbral Adaptativo")
            
            # Resaltar anomalías
            if np.any(anomaly_flags):
                anomaly_indices = np.where(anomaly_flags)[0]
                ax.scatter([datetimes[i] for i in anomaly_indices], 
                           scores[anomaly_indices], 
                           color='red', s=100, marker='o', label="Anomalías")
            
            # Configurar ejes y leyendas
            ax.set_title("Historial de Detección de Anomalías")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Puntuación")
            ax.legend()
            ax.grid(True)
            
            # Format the x-axis with a more efficient approach
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # Optional: adjust date display if needed
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            return fig
            
        except ImportError as e:
            logger.error(f"Error al crear visualización: {str(e)}")
            return None
    
    def plot_feature_contributions(self, anomaly_result: AnomalyDetectionResult, 
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Visualiza las contribuciones de las características a la anomalía detectada.
        
        Args:
            anomaly_result: Resultado de detección de anomalía
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            
            # Verificar que haya contribuciones
            if not anomaly_result.feature_contributions:
                logger.warning("No hay contribuciones de características para visualizar.")
                return None
            
            # Extraer contribuciones
            features, scores = zip(*anomaly_result.feature_contributions)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=figsize)
            
            # Graficar contribuciones
            ax.barh(features, scores, color='skyblue')
            ax.set_xlabel("Contribución")
            ax.set_title("Contribuciones de Características a la Anomalía")
            plt.tight_layout()
            
            return fig
            
        except ImportError as e:
            logger.error(f"Error al crear visualización: {str(e)}")
            return None


if __name__ == "__main__":
    # Ejemplo de uso del sistema de detección de anomalías
    # Generar datos de ejemplo
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 5))
    anomalous_data = np.random.normal(5, 1, (10, 5))
    data = np.vstack([normal_data, anomalous_data])

    # Inicializar detector
    detector = AnomalyDetector()
    
    # Entrenar detector con datos normales
    detector.fit(normal_data, feature_names=[f"feature_{i}" for i in range(5)])
    
    # Detectar anomalías en un lote de datos
    results = detector.process_batch(data, explain_anomalies=True)
    
    # Mostrar resultados
    for result in results:
        print(result)
    
    # Visualizar historial de anomalías
    fig = detector.plot_anomaly_history()
    if fig:
        plt.show()
    
    # Visualizar contribuciones de características para la primera anomalía detectada
    anomalies = [res for res in results if res.is_anomaly]
    if anomalies:
        fig = detector.plot_feature_contributions(anomalies[0])
        if fig:
            plt.show()
