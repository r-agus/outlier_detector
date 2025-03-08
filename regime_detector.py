#!/usr/bin/env python3
"""
Módulo para la detección de regímenes operacionales.

Este módulo implementa algoritmos para identificar diferentes modos de operación
o regímenes en los datos, lo que permite adaptar la detección de anomalías al
contexto operativo actual del sistema.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import deque
import time
import logging
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime

from config import config

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('regime_detector')

class BaseRegimeDetector(ABC):
    """
    Clase base abstracta para todos los detectores de régimen.
    Define la interfaz común que deben implementar todas las estrategias de detección.
    """
    
    def __init__(self, name: str = "base_regime_detector", config_override: Dict = None):
        """
        Inicialización del detector de régimen base.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica que sobrescribe la configuración global
        """
        self.name = name
        self.config_override = config_override if config_override else {}
        
        # Usar configuración centralizada
        self.detector_config = config.regime_detector
        self.current_regime = self.detector_config.default_regime
        self.previous_regime = self.detector_config.default_regime
        
        # Aplicar sobrescritura si se proporciona
        if self.config_override:
            for key, value in self.config_override.items():
                if hasattr(self.detector_config, key):
                    setattr(self.detector_config, key, value)
        
        # Historia de regímenes y timestamps asociados
        self.regime_history = deque(maxlen=1000)  
        self.timestamp_history = deque(maxlen=1000)
        self.last_change_time = time.time()
        
        # Callbacks para notificar cambios de régimen a otros componentes
        self.regime_change_callbacks = []
        
        # Add a flag to track if detector has been trained
        self.is_fitted = False
        
        # Add reference data statistics
        self.reference_stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "n_samples": 0
        }
        
        logger.info(f"Inicializando detector de régimen: {self.name}")
    
    def add_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Añade un callback que se ejecutará cuando cambie el régimen.
        
        Args:
            callback: Función que recibe (nuevo_régimen, régimen_anterior)
        """
        self.regime_change_callbacks.append(callback)
    
    def notify_regime_change(self, new_regime: str, old_regime: str) -> None:
        """
        Notifica a los callbacks registrados sobre un cambio de régimen.
        
        Args:
            new_regime: Nuevo régimen detectado
            old_regime: Régimen anterior
        """
        for callback in self.regime_change_callbacks:
            try:
                callback(new_regime, old_regime)
            except Exception as e:
                logger.error(f"Error en callback de cambio de régimen: {str(e)}")
    
    @abstractmethod
    def update(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series]) -> str:
        """
        Actualiza la detección de régimen con nuevos datos.
        
        Args:
            data_point: Nuevo punto de datos
            
        Returns:
            Régimen actual detectado
        """
        pass
    
    def get_current_regime(self) -> str:
        """
        Obtiene el régimen actual.
        
        Returns:
            Nombre del régimen actual
        """
        return self.current_regime
    
    def _check_and_update_regime(self, detected_regime: str) -> str:
        """
        Verifica si el régimen ha cambiado y actualiza el estado si es necesario.
        Incluye lógica para evitar oscilaciones rápidas entre regímenes.
        
        Args:
            detected_regime: Régimen detectado en la última iteración
            
        Returns:
            Régimen final después de aplicar estabilidad
        """
        now = time.time()
        min_duration = self.detector_config.min_regime_duration
        
        # Always add the current detection to history for better tracking
        self.regime_history.append(detected_regime)
        self.timestamp_history.append(now)
        
        # Si el régimen detectado es diferente al actual
        if detected_regime != self.current_regime:
            # Verificar si ha pasado suficiente tiempo desde el último cambio
            if now - self.last_change_time >= min_duration:
                old_regime = self.current_regime
                self.previous_regime = old_regime
                self.current_regime = detected_regime
                self.last_change_time = now
                
                # Notificar cambio
                self.notify_regime_change(detected_regime, old_regime)
                logger.info(f"Cambio de régimen: {old_regime} -> {detected_regime}")
            else:
                # No ha pasado suficiente tiempo, mantener régimen actual
                logger.debug(f"Régimen {detected_regime} detectado pero manteniendo {self.current_regime} por estabilidad")
        
        return self.current_regime
    
    def fit(self, training_data: np.ndarray) -> None:
        """
        Entrena el detector de régimen con datos históricos.
        Método base que implementa validaciones y cálculo de estadísticas básicas.
        
        Args:
            training_data: Datos históricos para entrenamiento
            
        Returns:
            None
            
        Raises:
            ValueError: Si los datos de entrenamiento son inválidos
        """
        # Validate training data
        if training_data is None:
            logger.warning(f"No training data provided to {self.name}")
            return
            
        if not isinstance(training_data, np.ndarray):
            try:
                training_data = np.array(training_data)
            except Exception as e:
                logger.error(f"Failed to convert training_data to numpy array: {str(e)}")
                raise ValueError("Training data must be convertible to numpy array")
            
        if len(training_data) == 0:
            logger.warning(f"Empty training data provided to {self.name}")
            return
            
        try:
            # Flatten 3D+ data to 2D for simpler processing
            if training_data.ndim > 2:
                original_shape = training_data.shape
                n_samples = original_shape[0]
                training_data = training_data.reshape(n_samples, -1)
                logger.info(f"Reshaped training data from {original_shape} to {training_data.shape}")
                
            # Calculate basic statistics
            self.reference_stats["mean"] = np.mean(training_data, axis=0)
            self.reference_stats["std"] = np.std(training_data, axis=0)
            self.reference_stats["min"] = np.min(training_data, axis=0)
            self.reference_stats["max"] = np.max(training_data, axis=0)
            self.reference_stats["n_samples"] = len(training_data)
            
            # Add initial data points to data window for immediate regime detection
            if hasattr(self, 'data_window') and self.data_window is not None:
                # Only take a subset if dataset is large
                n_samples_to_store = min(len(training_data), self.detector_config.window_size)
                for i in range(n_samples_to_store):
                    self.data_window.append(training_data[i])
            
            # Mark as fitted
            self.is_fitted = True
            
            logger.info(f"Base training completed for {self.name} with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error during base training for {self.name}: {str(e)}")
            raise
        
        return self
    
    def get_regime_duration(self) -> float:
        """
        Calcula cuánto tiempo lleva el sistema en el régimen actual.
        
        Returns:
            Duración en segundos
        """
        return time.time() - self.last_change_time
    
    def get_regime_history(self) -> List[Tuple[float, str]]:
        """
        Obtiene el historial de regímenes con sus timestamps.
        
        Returns:
            Lista de tuplas (timestamp, nombre_régimen)
        """
        return list(zip(self.timestamp_history, self.regime_history))
    
    def predict_next_regime(self) -> Optional[str]:
        """
        Intenta predecir el próximo régimen basado en patrones históricos.
        Método base que puede ser sobrescrito por implementaciones específicas.
        
        Returns:
            Nombre del régimen predicho o None si no se puede predecir
        """
        # Implementación por defecto: devuelve el régimen actual
        return self.current_regime


class StatisticalRegimeDetector(BaseRegimeDetector):
    """
    Detector de régimen basado en estadísticas de los datos.
    Utiliza umbrales estadísticos para determinar los regímenes basado en niveles de actividad.
    """
    
    def __init__(self, name: str = "statistical_regime_detector", 
                 config_override: Dict = None,
                 regimes: Dict[str, Dict[str, float]] = None):
        """
        Inicializa el detector de régimen estadístico.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica para sobrescribir configuración global
            regimes: Diccionario de regímenes con sus límites estadísticos
        """
        super().__init__(name, config_override)
        
        # Usar regímenes de la configuración central o los proporcionados explícitamente
        self.regimes = regimes if regimes is not None else self.detector_config.statistical_regimes
        
        # Ventana para estadísticas desde la configuración centralizada
        self.window_size = self.detector_config.window_size
        self.data_window = deque(maxlen=self.window_size)
        
        # Estabilidad de régimen de la configuración
        self.min_regime_duration = self.detector_config.min_regime_duration
        
        # Verificar y logear la configuración
        logger.info(f"Detector estadístico de régimen inicializado con {len(self.regimes)} regímenes: {list(self.regimes.keys())}")
        logger.debug(f"Umbrales de regímenes: {self.regimes}")
    
    def update(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series]) -> str:
        """
        # Verificar y logear la configuración
        logger.info(f"Detector estadístico de régimen inicializado con {len(self.regimes)} regímenes: {list(self.regimes.keys())}")
        logger.debug(f"Umbrales de regímenes: {self.regimes}")
    
    def update(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series]) -> str:
        Returns:
            Régimen actual detectado
        """
        # Convertir a ndarray si es necesario
        if isinstance(data_point, pd.DataFrame) or isinstance(data_point, pd.Series):
            data_point = data_point.values
        
        # Si es un lote de datos, procesar cada punto
        if data_point.ndim > 1 and data_point.shape[0] > 1:
            # Procesar todo el lote
        Determina el régimen basado en estadísticas calculadas.
        
        Args:
            mean_value: Valor medio calculado
            std_value: Desviación estándar calculada
            
        else:
            # Un solo punto
            if data_point.ndim > 1:
                data_point = data_point.flatten()
            self.data_window.append(data_point)
        
        # Calcular estadísticas solo si hay suficientes datos
        if len(self.data_window) < 3:
            return self.current_regime
        
        # Convertir a array para cálculos estadísticos
        window_data = np.array(self.data_window)
        
        # Calcular estadísticas
        data_mean = np.mean(np.abs(window_data))  # Media absoluta para todos los canales
        data_std = np.std(window_data)  # Desviación estándar
        
        # Determinar régimen basado en estadísticas
        detected_regime = self._determine_regime_from_stats(data_mean, data_std)
        
        # Verificar si hay que actualizar y notificar cambios
        return self._check_and_update_regime(detected_regime)
    
    def _determine_regime_from_stats(self, mean_value: float, std_value: float) -> str:
        """
        Determina el régimen basado en estadísticas calculadas.
        
        Args:
            mean_value: Valor medio calculado
            std_value: Desviación estándar calculada
            
        Returns:
            Diccionario con estadísticas (mean, std, min, max)
        """
            Nombre del régimen detectado
        """
        # Log current statistics for debugging
        logger.debug(f"Estadísticas - Media: {mean_value:.4f}, Desviación: {std_value:.4f}")
        
        # FIXED: Improve regime detection with more robust logic
        for regime_name, thresholds in self.regimes.items():
            matches = True
            
            # Check mean thresholds
            if "min_mean" in thresholds and mean_value < thresholds["min_mean"]:
                matches = False
            if "max_mean" in thresholds and mean_value > thresholds["max_mean"]:
                matches = False
                
            # Check std thresholds
            if "min_std" in thresholds and std_value < thresholds["min_std"]:
                matches = False
            if "max_std" in thresholds and std_value > thresholds["max_std"]:
                matches = False
            
            if matches:
                return regime_name
        
        # FIXED: More robust fallback - prefer current regime for stability 
        # if no match is found rather than always choosing default
        logger.debug(f"No matching regime found, maintaining {self.current_regime}")
        return self.current_regime
class TimeBasedRegimeDetector(BaseRegimeDetector):
    """
    Detector de régimen basado en patrones temporales.
    Útil para sistemas donde los regímenes siguen patrones diarios, semanales, etc.
    """
    
    def __init__(self, name: str = "time_based_regime_detector", 
                 config_override: Dict = None):
        """
        Inicializa el detector de régimen basado en tiempo.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Usar configuración centralizada para regímenes por hora
        self.hour_regimes = self.detector_config.hour_regimes
        
        logger.info(f"Detector de régimen basado en tiempo inicializado con {len(set(self.hour_regimes.values()))} regímenes posibles")
    
    def update(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series] = None) -> str:
        """
        Actualiza la detección de régimen basada en la hora actual.
        Nota: Esta implementación ignora los datos, usa solo la hora del sistema.
        
        Args:
            data_point: No utilizado, incluido para compatibilidad con la interfaz
            
        Returns:
            Régimen actual detectado
        """
        # Obtener hora actual
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Determinar régimen basado en la hora
        detected_regime = self.hour_regimes.get(current_hour, self.detector_config.default_regime)
        
        # Verificar si hay que actualizar y notificar cambios
        return self._check_and_update_regime(detected_regime)


class ClusteringRegimeDetector(BaseRegimeDetector):
    """
    Detector de régimen basado en clustering.
    Agrupa los datos en diferentes modos de operación mediante algoritmos de clustering.
    """
    
    def __init__(self, name: str = "clustering_regime_detector", 
                 config_override: Dict = None,
                 n_clusters: int = 3):
        """
        Inicializa el detector de régimen basado en clustering.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica para sobrescribir configuración global
            n_clusters: Número de clusters (regímenes) a detectar
        """
        super().__init__(name, config_override)
        
        # Parámetros de clustering
        self.n_clusters = n_clusters
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_fitted = False
        
        # Mapeo de clusters a nombres de regímenes desde configuración
        self.cluster_to_regime = clustering_config.get("cluster_regime_mapping", {})
        
        # If no mapping provided, create a default mapping
        if not self.cluster_to_regime:
            self.cluster_to_regime = {
                0: "low_activity",
                1: "normal",
                2: "high_activity"
            }
        
        # Ventana de datos para adaptación continua
        self.window_size = self.detector_config.window_size
        self.data_window = deque(maxlen=self.window_size)
        
        # Centroides de los clusters
        self.centroids = None
        
        # Período de reentrenamiento de la configuración
        self.refit_interval = clustering_config.get("refit_interval", 1000)
        self.points_since_last_fit = 0
        
        logger.info(f"Detector de régimen por clustering inicializado con {self.n_clusters} clusters")
    
    def fit(self, training_data: np.ndarray) -> None:
        """
        Entrena el modelo de clustering con datos históricos.
        
        Args:
            training_data: Datos históricos para entrenamiento
        """
        if len(training_data) < self.n_clusters:
            logger.warning(f"Insuficientes datos para clustering: {len(training_data)} puntos, {self.n_clusters} clusters")
            return
            
        try:
            # Reshape si es necesario
            if training_data.ndim > 2:
                n_samples = training_data.shape[0]
                training_data = training_data.reshape(n_samples, -1)
            
            # Entrenar modelo
            self.cluster_model.fit(training_data)
            self.centroids = self.cluster_model.cluster_centers_
            self.is_fitted = True
            logger.info(f"Modelo de clustering entrenado con {len(training_data)} puntos")
        except Exception as e:
            logger.error(f"Error al entrenar modelo de clustering: {str(e)}")
    
    def update(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series]) -> str:
        """
        Actualiza la detección de régimen con nuevos datos.
        
        Args:
            data_point: Nuevo punto de datos
            
        Returns:
            Régimen actual detectado
        """
        # Convertir a ndarray si es necesario
        if isinstance(data_point, pd.DataFrame) or isinstance(data_point, pd.Series):
            data_point = data_point.values
        
        # Procesar punto o lote
        if data_point.ndim > 1 and data_point.shape[0] > 1:
            # Si es un lote, usar solo el último punto para la decisión actual
            point = data_point[-1]
            # Pero añadir todos a la ventana
            for p in data_point:
                self.data_window.append(p)
                self.points_since_last_fit += 1
        else:
            # Un solo punto
            point = data_point.flatten() if data_point.ndim > 1 else data_point
            self.data_window.append(point)
            self.points_since_last_fit += 1
        
        # Si no está entrenado y hay suficientes datos en la ventana, entrenar
        if not self.is_fitted and len(self.data_window) >= max(self.n_clusters * 2, 10):
            window_data = np.array(self.data_window)
            self.fit(window_data)
        
        # Si sigue sin estar entrenado, usar régimen por defecto
        if not self.is_fitted:
            return self._check_and_update_regime(self.detector_config.default_regime)
        
        # Determinar cluster del punto actual
        try:
            # Reshape para predecir
            point_reshaped = point.reshape(1, -1)
            cluster = self.cluster_model.predict(point_reshaped)[0]
            
            # Convertir cluster a nombre de régimen
            detected_regime = self.cluster_to_regime.get(
                int(cluster), self.detector_config.default_regime
            )
            
            logger.debug(f"Cluster {cluster} mapeado a régimen: {detected_regime}")
            
            # Verificar si es momento de reentrenar
            if self.points_since_last_fit >= self.refit_interval:
                logger.debug("Reentrenando modelo de clustering con datos recientes")
                window_data = np.array(self.data_window)
                self.fit(window_data)
                self.points_since_last_fit = 0
            
            # Verificar si hay que actualizar y notificar cambios
            return self._check_and_update_regime(detected_regime)
            
        except Exception as e:
            logger.error(f"Error al predecir cluster: {str(e)}")
            return self._check_and_update_regime(self.detector_config.default_regime)


class HybridRegimeDetector(BaseRegimeDetector):
    """
    Detector de régimen híbrido que combina múltiples estrategias.
    Utiliza detección estadística, temporal y de clustering, con votación ponderada.
    """
    
    def __init__(self, name: str = "hybrid_regime_detector", config_override: Dict = None):
        """
        Inicializa el detector de régimen híbrido.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Crear detectores individuales
        self.statistical_detector = StatisticalRegimeDetector("statistical")
        self.time_based_detector = TimeBasedRegimeDetector("time_based")
        self.clustering_detector = ClusteringRegimeDetector("clustering")
        
        # Lista de detectores activos
        self.detectors = [
            self.statistical_detector,
            self.time_based_detector,
            self.clustering_detector
        ]
        
        # Obtener pesos para cada detector desde la configuración
        hybrid_weights = self.detector_config.hybrid_weights
        self.weights = {
            "statistical": hybrid_weights.get("statistical", 1.0),
            "time_based": hybrid_weights.get("time_based", 0.8),
            "clustering": hybrid_weights.get("clustering", 0.6)
        }
        
        # Estabilidad de régimen (evitar oscilaciones)
        self.stability_window = deque(maxlen=5)  # Últimas 5 detecciones
        self.min_confidence = 0.6  # Confianza mínima para cambiar régimen
        
        # Create a data window specifically for this detector to avoid early reference errors
        self._data_window = deque(maxlen=self.detector_config.window_size)
        
        logger.info(f"Detector de régimen híbrido inicializado con pesos: {self.weights}")
    
    def fit(self, training_data: np.ndarray) -> None:
        """
        Entrena todos los subdetectores con datos históricos.
        
        Args:
            training_data: Datos históricos para entrenamiento
        """
        # Initialize the data window with training data for early regime detection
        if training_data is not None and len(training_data) > 0:
            for point in training_data[-self.detector_config.window_size:]:
                self._data_window.append(point)
        
        # Train individual detectors
        for detector in self.detectors:
            try:
                detector.fit(training_data)
            except Exception as e:
                logger.error(f"Error al entrenar {detector.name}: {str(e)}")
    
    def update(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series]) -> str:
        """
        Actualiza la detección de régimen combinando múltiples estrategias.
        
        Args:
            data_point: Nuevo punto de datos
            
        Returns:
            Régimen actual detectado
        """
        # Preprocess data point and add to local data window
        if isinstance(data_point, pd.DataFrame) or isinstance(data_point, pd.Series):
            data_point_array = data_point.values
        else:
            data_point_array = data_point
        
        # Add to our local window to ensure it exists before early regime detection
        if data_point_array.ndim > 1 and data_point_array.shape[0] > 1:
            # If batch, add the last point
            self._data_window.append(data_point_array[-1])
        else:
            # Single point
            flat_point = data_point_array.flatten() if data_point_array.ndim > 1 else data_point_array
            self._data_window.append(flat_point)
            
        # Early regime detection with minimal data
        if len(self.data_window) < 5:  # If very little data is available
            # Use just statistical detector for immediate feedback
            try:
                regime = self.statistical_detector.update(data_point)
                return self._check_and_update_regime(regime)
            except Exception as e:
                logger.error(f"Error in early regime detection: {str(e)}")
                return self.current_regime
        
        # Obtener predicciones de cada detector
        regimes = {}
        
        # Actualizar cada detector y obtener su régimen
        for detector in self.detectors:
            try:
                detector_name = detector.name.split('_')[0]  # Obtener parte base del nombre
                regime = detector.update(data_point)
                
                # Inicializar contador si no existe
                if regime not in regimes:
                    regimes[regime] = 0
                
                # Añadir voto ponderado
                regimes[regime] += self.weights.get(detector_name, 1.0)
                logger.debug(f"Detector {detector_name} votó por régimen: {regime} con peso {self.weights.get(detector_name, 1.0)}")
                
            except Exception as e:
                logger.error(f"Error al actualizar detector {detector.name}: {str(e)}")
        
        # Si no hay votos, mantener régimen actual
        if not regimes:
            return self.current_regime
        
        # Determinar régimen ganador
        sorted_regimes = sorted(regimes.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_regimes[0][0]
        
        # Calculate confidence level
        total_weight = sum(regimes.values())
        confidence = sorted_regimes[0][1] / total_weight if total_weight > 0 else 0
        logger.debug(f"Régimen ganador: {winner} con confianza {confidence:.2f}")
        
        # Add to stability window
        self.stability_window.append(winner)
        
        # Check if winning regime appears enough times in stability window
        if len(self.stability_window) >= 3:
            counts = {}
            for r in self.stability_window:
                if r not in counts:
                    counts[r] = 0
                counts[r] += 1
            
            most_common = max(counts.items(), key=lambda x: x[1])
            stability_confidence = most_common[1] / len(self.stability_window)
            
            # If stable enough, use most common regime
            if stability_confidence >= 0.6:  # At least 60% agreement
                winner = most_common[0]
                logger.debug(f"Usando régimen estable: {winner} con confianza {stability_confidence:.2f}")
            
        # Verificar si hay que actualizar y notificar cambios
        return self._check_and_update_regime(winner)
    
    @property
    def data_window(self):
        """
        Property that provides access to the data window.
        Returns either the local window or the statistical detector's window.
        """
        if hasattr(self, '_data_window') and self._data_window:
            return self._data_window
        elif hasattr(self.statistical_detector, 'data_window'):
            return self.statistical_detector.data_window
        else:
            # Fallback to avoid errors
            if not hasattr(self, '_data_window'):
                self._data_window = deque(maxlen=self.detector_config.window_size)
            return self._data_window


# Additional utility functions to help with regime detection

def detect_concept_drift(old_data: np.ndarray, new_data: np.ndarray, 
                        alpha: float = 0.05) -> bool:
    """
    Detects if there's a significant distribution change between old and new data.
    
    Args:
        old_data: Reference data (historical)
        new_data: Current data
        alpha: Significance level for the test
        
    Returns:
        True if drift detected, False otherwise
    """
    # If data is multidimensional, perform test on flattened distributions
    old_flat = old_data.flatten() if old_data.ndim > 1 else old_data
    new_flat = new_data.flatten() if new_data.ndim > 1 else new_data
    
    try:
        # Perform two-sample Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(old_flat, new_flat)
        return p_value < alpha
    except Exception as e:
        logger.error(f"Error in concept drift detection: {str(e)}")
        return False


def plot_regime_transitions(detector: BaseRegimeDetector, figsize=(12, 6)) -> plt.Figure:
    """
    Visualizes regime transitions over time.
    
    Args:
        detector: Regime detector with history
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    try:
        history = detector.get_regime_history()
        if not history:
            return None
            
        times, regimes = zip(*history)
        
        # Convert timestamps to datetimes
        datetimes = [datetime.fromtimestamp(t) for t in times]
        
        # Get unique regimes and assign colors
        unique_regimes = sorted(set(regimes))
        regime_to_num = {regime: i for i, regime in enumerate(unique_regimes)}
        
        # Convert regimes to numeric values for plotting
        regime_values = [regime_to_num[r] for r in regimes]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot regime transitions
        ax.plot(datetimes, regime_values, 'b-', drawstyle='steps-post')
        ax.scatter(datetimes, regime_values, c='blue', alpha=0.5)
        
        # Set y-ticks to regime names
        ax.set_yticks(range(len(unique_regimes)))
        ax.set_yticklabels(unique_regimes)
        
        # Format plot
        ax.set_title("Transitions Between Operating Regimes")
        ax.set_xlabel("Time")
        ax.set_ylabel("Regime")
        ax.grid(True, alpha=0.3)
        
        # Format date axis
        plt.gcf().autofmt_xdate()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting regime transitions: {str(e)}")
        return None


def plot_detector_performance(ax, detector, title, ground_truth, n_points_per_regime):
    """Helper function to plot detector performance with consistent formatting"""
    history = detector.get_regime_history()
    
    if not history:
        ax.text(0.5, 0.5, "No hay datos de historial disponibles", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Extract data
    regimes = [h[1] for h in history]
    
    # Define consistent colors for regimes
    regime_colors = {
        'low_activity': 'blue',
        'normal': 'green',
        'high_activity': 'red'
    }
    
    # Map regimes to numeric values for plotting
    regime_values = []
    for r in regimes:
        if r == 'low_activity':
            regime_values.append(0)
        elif r == 'normal':
            regime_values.append(1)
        elif r == 'high_activity':
            regime_values.append(2)
        else:
            regime_values.append(-1)  # Unknown regime
    
    # Plot detected regimes
    ax.step(range(len(regime_values)), regime_values, where='post', 
           linewidth=2, label="Régimen Detectado")
    
    # Add vertical lines for true regime changes
    ax.axvline(x=n_points_per_regime, color='k', linestyle='--', alpha=0.7)
    ax.axvline(x=n_points_per_regime*2, color='k', linestyle='--', alpha=0.7)
    
    # Add background colors for true regimes
    ax.axvspan(0, n_points_per_regime, alpha=0.2, color=regime_colors['low_activity'])
    ax.axvspan(n_points_per_regime, n_points_per_regime*2, alpha=0.2, color=regime_colors['normal'])
    ax.axvspan(n_points_per_regime*2, len(regime_values), alpha=0.2, color=regime_colors['high_activity'])
    
    # Calculate accuracy
    correct = 0
    for i, regime in enumerate(regimes):
        if i < len(ground_truth) and regime == ground_truth[i]:
            correct += 1
    
    accuracy = correct / len(ground_truth) if ground_truth else 0
    
    # Set labels and title
    ax.set_title(f"{title} - Precisión: {accuracy:.2%}")
    ax.set_xlabel("Índice de Muestra")
    ax.set_ylabel("Régimen")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Baja', 'Normal', 'Alta'])
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    """
    Demonstrate the behavior of different regime detectors with synthetic data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import time
    from datetime import datetime, timedelta
    
    print("=" * 70)
    print("DEMOSTRACIÓN DE DETECCIÓN DE REGÍMENES OPERACIONALES")
    print("=" * 70)
    
    # Configure visualization
    plt.style.use('ggplot')
    np.random.seed(42)
    
    # =========================================================================
    # 1. DATA GENERATION - Create synthetic data with clear regime transitions
    # =========================================================================
    print("\n1. Generando datos sintéticos con cambios de régimen...")
    
    # Time parameters
    n_points_per_regime = 300
    n_features = 3
    total_points = n_points_per_regime * 3
    
    # Generate timestamps (one point every minute)
    base_time = datetime.now() - timedelta(minutes=total_points)
    timestamps = [base_time + timedelta(minutes=i) for i in range(total_points)]
    time_values = [time.mktime(ts.timetuple()) for ts in timestamps]
    
    # Regime 1: Low activity (low values, low variance)
    print("  - Régimen 1: Baja actividad (valores bajos, varianza baja)")
    low_activity_data = np.random.normal(0, 0.2, (n_points_per_regime, n_features))
    
    # Regime 2: Normal operation (medium values, medium variance)
    print("  - Régimen 2: Operación normal (valores medios, varianza media)")
    normal_data = np.random.normal(1, 0.5, (n_points_per_regime, n_features))
    
    # Regime 3: High activity (high values, high variance)
    print("  - Régimen 3: Alta actividad (valores altos, varianza alta)")
    high_activity_data = np.random.normal(3, 1.2, (n_points_per_regime, n_features))
    
    # Combine all regimes into one dataset
    all_data = np.vstack([low_activity_data, normal_data, high_activity_data])
    
    # Ground truth regime labels
    ground_truth = ['low_activity'] * n_points_per_regime + \
                  ['normal'] * n_points_per_regime + \
                  ['high_activity'] * n_points_per_regime
    
    # =========================================================================
    # 2. DETECTOR INITIALIZATION - Create one of each regime detector type
    # =========================================================================
    print("\n2. Inicializando detectores de régimen...")
    
    # Statistical detector (based on statistical thresholds)
    print("  - Detector Estadístico: umbral basado en estadísticas")
    stat_detector = StatisticalRegimeDetector("statistical_demo", 
                                             regimes={
                                                 "low_activity": {
                                                     "max_mean": 0.5,
                                                     "max_std": 0.3
                                                 },
                                                 "normal": {
                                                     "min_mean": 0.5,
                                                     "max_mean": 2.0,
                                                     "min_std": 0.3,
                                                     "max_std": 0.8
                                                 },
                                                 "high_activity": {
                                                     "min_mean": 2.0,
                                                     "min_std": 0.8
                                                 }
                                             })
    
    # Clustering detector (based on K-means)
    print("  - Detector de Clustering: agrupamiento con K-means")
    cluster_detector = ClusteringRegimeDetector("clustering_demo", n_clusters=3)
    
    # Time-based detector (simulated by mapping points to time ranges)
    print("  - Detector Temporal: régimen basado en hora/fecha")
    time_detector = TimeBasedRegimeDetector("time_demo")
    
    # Hybrid detector (combines all previous detectors)
    print("  - Detector Híbrido: combinación ponderada de detectores")
    hybrid_detector = HybridRegimeDetector("hybrid_demo")
    
    # List all detectors for processing
    detectors = [
        stat_detector,
        cluster_detector,
        time_detector,
        hybrid_detector
    ]
    
    # =========================================================================
    # 3. TRAINING - Train the detectors with sample data
    # =========================================================================
    print("\n3. Entrenando detectores...")
    
    # Training data (a small sample from each regime)
    training_samples = n_points_per_regime // 10
    training_data = np.vstack([
        low_activity_data[:training_samples],
        normal_data[:training_samples],
        high_activity_data[:training_samples]
    ])
    
    # Train each detector
    for detector in detectors:
        print(f"  - Entrenando detector: {detector.name}")
        detector.fit(training_data)
    
    # =========================================================================
    # 4. DETECTION - Run each regime detector on the full dataset
    # =========================================================================
    print("\n4. Ejecutando detección de régimen...")
    
    # For testing - reduce min regime duration to 0 to ensure detection works
    for detector in detectors:
        detector.detector_config.min_regime_duration = 0
    
    # Store predictions from each detector
    predictions = {}
    regimes_by_detector = {}
    
    # Process data with each detector
    for detector in detectors:
        detector_name = detector.name
        print(f"  - Procesando con {detector_name}")
        
        # Reset detector state
        detector.current_regime = detector.detector_config.default_regime
        detector.regime_history.clear()
        detector.timestamp_history.clear()
        
        # Track regime changes
        regime_changes = []
        
        # Process data point by point to ensure consistent history tracking
        for i in range(len(all_data)):
            point = all_data[i]
            point_time = time_values[i]
            regime_idx = i // n_points_per_regime
            
            # For time-based detector, set appropriate time-based regime
            if isinstance(detector, TimeBasedRegimeDetector):
                # Map each point to an appropriate hour based on its regime
                fake_hours = np.array([3, 12, 20])  # Map to hours for each regime
                fake_hour = fake_hours[min(regime_idx, 2)]
                
                # Create a timestamp with the simulated hour
                fake_time = datetime(2023, 1, 1, int(fake_hour), 0, 0)
                
                # Process the point with timestamp override
                detector.update(point)
            else:
                # Process one point at a time - no batches
                detector.update(point)
                
        # Collect regime history for visualization
        regimes_by_detector[detector_name] = detector.get_regime_history()
        
        # Print detection information
        history = detector.get_regime_history()
        if history:
            regimes = [r[1] for r in history]
            transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
            print(f"    * Puntos procesados: {len(regimes)}/{len(all_data)}")
            print(f"    * Transiciones detectadas: {transitions}")
    
    # =========================================================================
    # 5. VISUALIZATION - Plot and compare detector behaviors
    # =========================================================================
    print("\n5. Visualizando resultados...")
    
    # Create comparison figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 2, 2])
    
    # Plot 1: Show the data
    ax_data = fig.add_subplot(gs[0, :])
    ax_data.set_title("Datos Sintéticos con 3 Regímenes", fontsize=14)
    
    # Plot each feature with semi-transparency to show density
    for feature in range(n_features):
        ax_data.plot(all_data[:, feature], alpha=0.4, label=f'Característica {feature+1}')
    
    # Add vertical lines to show true regime transitions
    ax_data.axvline(x=n_points_per_regime, color='k', linestyle='--', alpha=0.7)
    ax_data.axvline(x=n_points_per_regime*2, color='k', linestyle='--', alpha=0.7)
    
    # Add text labels for true regimes
    y_pos = ax_data.get_ylim()[1] * 0.9
    ax_data.text(n_points_per_regime*0.5, y_pos, "Baja Actividad", ha='center')
    ax_data.text(n_points_per_regime*1.5, y_pos, "Normal", ha='center')
    ax_data.text(n_points_per_regime*2.5, y_pos, "Alta Actividad", ha='center')
    
    ax_data.set_xlabel("Índice de Muestra")
    ax_data.set_ylabel("Valor")
    ax_data.legend(loc='upper right')
    ax_data.grid(True, alpha=0.3)
    
    # Plot 2: Statistical detector performance
    ax_stat = fig.add_subplot(gs[1, 0])
    plot_detector_performance(ax_stat, stat_detector, "Detector Estadístico", 
                            ground_truth, n_points_per_regime)
    
    # Plot 3: Clustering detector performance
    ax_cluster = fig.add_subplot(gs[1, 1])
    plot_detector_performance(ax_cluster, cluster_detector, "Detector de Clustering", 
                             ground_truth, n_points_per_regime)
    
    # Plot 4: Time-based detector performance
    ax_time = fig.add_subplot(gs[2, 0])
    plot_detector_performance(ax_time, time_detector, "Detector Basado en Tiempo", 
                           ground_truth, n_points_per_regime)
    
    # Plot 5: Hybrid detector performance
    ax_hybrid = fig.add_subplot(gs[2, 1])
    plot_detector_performance(ax_hybrid, hybrid_detector, "Detector Híbrido", 
                            ground_truth, n_points_per_regime)
    
    plt.tight_layout()
    
    # Summary plot - compare all detectors regime changes
    plt.figure(figsize=(15, 8))
    #plt.title("Comparación de Detección de Regímenes", fontsize=14)
    
    # Define regime mapping for consistent coloring
    regime_colors = {
        'low_activity': 'blue',
        'normal': 'green',
        'high_activity': 'red',
        'default': 'purple'
    }
    
    # For each detector, calculate detection accuracy
    accuracy_stats = {}
    for i, detector in enumerate(detectors):
        detector_name = detector.name
        history = detector.get_regime_history()
        
        # Skip if no history
        if not history:
            continue
            
        # Extract regimes and convert to numeric for plotting
        regimes = [r[1] for r in history]
        
        # Calculate agreement with ground truth
        matches = sum(1 for i, r in enumerate(regimes) if i < len(ground_truth) and r == ground_truth[i])
        accuracy = matches / len(ground_truth) if ground_truth else 0
        accuracy_stats[detector_name] = accuracy
        
        # Plot regimes for this detector
        plt.subplot(len(detectors), 1, i+1)
        
        # Create a continuous color map for the detector's regimes
        colors = []
        for regime in regimes:
            colors.append(regime_colors.get(regime, regime_colors['default']))
            
        # Create step plot with colored segments
        indices = list(range(len(regimes)))
        plt.step(indices, [i] * len(indices), where='post', color='black', alpha=0.2)
        
        # Add colored background segments for each regime
        prev_idx = 0
        prev_regime = regimes[0]
        for idx, regime in enumerate(regimes[1:], 1):
            if regime != prev_regime:
                plt.axvspan(prev_idx, idx, alpha=0.3, color=regime_colors.get(prev_regime, regime_colors['default']))
                prev_idx = idx
                prev_regime = regime
                
        # Last segment
        plt.axvspan(prev_idx, len(regimes), alpha=0.3, color=regime_colors.get(prev_regime, regime_colors['default']))
        
        # Add vertical lines for true regime changes
        plt.axvline(x=n_points_per_regime, color='k', linestyle='--', alpha=0.7)
        plt.axvline(x=n_points_per_regime*2, color='k', linestyle='--', alpha=0.7)
        
        plt.title(f"{detector_name} - Precisión: {accuracy:.2%}")
        plt.yticks([])  # Hide y-axis
        
        if i == len(detectors) - 1:
            plt.xlabel("Índice de Muestra")
    
    # Create legend
    handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.3) 
              for regime, color in regime_colors.items()]
    labels = list(regime_colors.keys())
    plt.figlegend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout()
    
    # =========================================================================
    # 6. RESULTS ANALYSIS - Print summary statistics
    # =========================================================================
    print("\n6. Análisis de resultados:")
    print("-" * 60)
    print(f"{'Detector':<20} | {'Precisión':<15} | {'Cambios Detectados':<20}")
    print("-" * 60)
    
    for detector in detectors:
        detector_name = detector.name
        history = detector.get_regime_history()
        
        # Skip if no history
        if not history or detector_name not in accuracy_stats:
            continue
            
        # Count transitions
        regimes = [r[1] for r in history]
        transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        
        # Get accuracy
        accuracy = accuracy_stats[detector_name]
        
        print(f"{detector_name:<20} | {accuracy:>13.2%} | {transitions:>20}")
    
    print("-" * 60)
    
    # Show plots
    print("\nMostrando visualizaciones... (cierre las ventanas de gráficos para terminar)")
    plt.show()