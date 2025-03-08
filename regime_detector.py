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
            hour: Hora (0-23)
            regime: Régimen a asignar
        """
        if hour < 0 or hour > 23:
            raise ValueError("La hora debe estar entre 0 y 23")
        
        self.hour_regimes[hour] = regime
        logger.info(f"Hora {hour} asignada al régimen '{regime}'")
    
    def set_weekday_modifier(self, weekday: int, modifier: str) -> None:
        """
        Establece el modificador de régimen para un día de la semana.
        
        Args:
            weekday: Día de la semana (0=lunes, 6=domingo)
            modifier: Modificador a asignar
        """
        if weekday < 0 or weekday > 6:
            raise ValueError("El día de la semana debe estar entre 0 (lunes) y 6 (domingo)")
        
        self.weekday_modifiers[weekday] = modifier
        logger.info(f"Día {weekday} asignado al modificador '{modifier}'")
    
    def add_special_regime(self, weekday: int, hour: int, regime: str) -> None:
        """
        Añade un régimen especial para una combinación específica de día y hora.
        
        Args:
            weekday: Día de la semana (0=lunes, 6=domingo)
            hour: Hora (0-23)
            regime: Régimen especial a asignar
        """
        self.special_regimes[(weekday, hour)] = regime
        logger.info(f"Régimen especial '{regime}' añadido para día {weekday}, hora {hour}")
            data_point: No utilizado, incluido para compatibilidad con la interfaz
    
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
        
        # Mapeo de clusters a nombres de regímenes
        self.cluster_to_regime = {}
        
        # Ventana de datos para adaptación continua
        self.window_size = config.adaptation.sliding_window.get("size", 100)
        self.data_window = deque(maxlen=self.window_size)
        
        # Centroides de los clusters
        self.centroids = None
        
        # Período de reentrenamiento
        self.refit_interval = 1000  # Puntos de datos entre reentrenamientos
        self.points_since_last_fit = 0
        
        logger.info(f"Detector de régimen por clustering inicializado con {n_clusters} clusters")
    
    def set_regime_names(self, cluster_to_regime: Dict[int, str]) -> None:
        """
        Establece nombres para los regímenes asociados a cada cluster.
        
        Args:
            cluster_to_regime: Mapeo de índices de cluster a nombres de régimen
        """
        self.cluster_to_regime = cluster_to_regime
        logger.info(f"Nombres de régimen establecidos: {cluster_to_regime}")
        # Obtener hora actual
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Determinar régimen basado en la hora
        detected_regime = self.hour_regimes.get(current_hour, self.detector_config.default_regime)
            else:
                train_data = data
            
            # Entrenar modelo
            self.cluster_model.fit(train_data)
            self.centroids = self.cluster_model.cluster_centers_
            self.is_fitted = True
            
            # Si no hay mapeo de cluster a régimen, crear uno predeterminado
            # ordenando los clusters por la magnitud de sus centroides
            if not self.cluster_to_regime:
                # Ordenar centroides por magnitud
                centroid_norms = np.linalg.norm(self.centroids, axis=1)
                sorted_indices = np.argsort(centroid_norms)
                
                regimes = ["low_activity", "normal", "high_activity"]
                if self.n_clusters <= len(regimes):
                    for i, idx in enumerate(sorted_indices):
                        if i < len(regimes):
                            self.cluster_to_regime[idx] = regimes[i]
                else:
                    # Si hay más clusters que nombres predefinidos
                    for i, idx in enumerate(sorted_indices):
                        if i == 0:
                            self.cluster_to_regime[idx] = "low_activity"
                        elif i == len(sorted_indices) - 1:
                            self.cluster_to_regime[idx] = "high_activity"
                        else:
                            self.cluster_to_regime[idx] = f"normal_{i}"
            
            logger.info(f"Modelo de clustering entrenado con {len(data)} puntos")
            self.points_since_last_fit = 0
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo de clustering: {str(e)}")
            raise
    
    def detect(self, data: np.ndarray) -> str:
        """
        Detecta el régimen actual basado en clustering.
        
        Args:
            data: Datos recientes para analizar
            
        Returns:
            Nombre del régimen detectado
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
            
            # Graficar centroides
            ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                      marker='X', s=200, c='black', label='Centroides')
            
            # Añadir etiquetas con nombres de régimen a los centroides
            for i, (x, y) in enumerate(centroids_2d):
                regime_name = self.cluster_to_regime.get(i, f"cluster_{i}")
                ax.annotate(regime_name, (x, y), 
                           textcoords="offset points", 
                           xytext=(0, 10), 
                           ha='center')
            
            ax.set_title("Clustering de Regímenes Operativos")
            ax.set_xlabel("Dimensión 1")
            ax.set_ylabel("Dimensión 2")
            ax.legend()
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error al visualizar clusters: {str(e)}")
            return None


class HybridRegimeDetector(BaseRegimeDetector):
    """
    Detector de régimen híbrido que combina múltiples estrategias.
    Utiliza detección estadística, temporal y de clustering, con votación.
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
        
        # Pesos para cada detector
        self.weights = {
            "statistical": 1.0,
            "time_based": 0.8,
            "clustering": 1.2
        }
        
        # Estabilidad de régimen (evitar oscilaciones)
        self.stability_window = deque(maxlen=5)  # Últimas 5 detecciones
        self.min_confidence = 0.6  # Confianza mínima para cambiar régimen
        
        logger.info(f"Detector de régimen híbrido inicializado")
    
    def set_detector_weight(self, detector_name: str, weight: float) -> None:
        """
        Establece el peso para un detector específico.
        
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
    
    def detect(self, data: np.ndarray) -> str:
        """
        Detecta el régimen usando un enfoque híbrido con votación ponderada.
        
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
            detector_name = detector.name
            weight = self.weights.get(detector_name, 0.0)
            
            if weight > 0:
                regime = detector.detect(data)
                if regime in regime_votes:
                    regime_votes[regime] += weight
                else:
                    regime_votes[regime] = weight
        
        # Sin predicciones, mantener régimen actual
        if not regime_votes:
            return self.current_regime
        
        # Encontrar el régimen con más votos
        total_weight = sum(regime_votes.values())
        winning_regime = max(regime_votes.items(), key=lambda x: x[1])
        confidence = winning_regime[1] / total_weight
        
        # Añadir a ventana de estabilidad
        self.stability_window.append(winning_regime[0])
        
        # Si el régimen ganador tiene confianza suficiente o 
        # ha sido consistente en las últimas detecciones
        if (confidence >= self.min_confidence or 
            self.stability_window.count(winning_regime[0]) >= len(self.stability_window) * 0.7):
            
            detected_regime = winning_regime[0]
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
    # Ejemplo de uso de detectores de régimen
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("Iniciando demostración de detectores de régimen...")
    
    # 1. Generar datos sintéticos para demostración
    np.random.seed(42)
    
    # Crear series temporales con diferentes regímenes
    # Régimen 1: Amplitud baja, poco ruido (0-200)
    # Régimen 2: Amplitud alta, ruido moderado (200-400)
    # Régimen 3: Amplitud media, ruido alto (400-600)
    points = 600
    time_values = np.linspace(0, 60, points)
    
    # Señal base
    base_signal = np.sin(time_values * 0.5)
    
    # Añadir características específicas de régimen
    signal = np.zeros_like(base_signal)
    
    # Régimen 1: Actividad baja (0-200)
    signal[:200] = 0.5 * base_signal[:200] + 0.1 * np.random.randn(200)
    
    # Régimen 2: Actividad alta (200-400)
    signal[200:400] = 2.0 * base_signal[200:400] + 0.3 * np.random.randn(200)
    
    # Régimen 3: Actividad normal con ruido (400-600)
    signal[400:600] = 1.0 * base_signal[400:600] + 0.8 * np.random.randn(200)
    
    # Crear puntos multidimensionales para pruebas más complejas
    # Añadimos una segunda dimensión correlacionada pero con diferente escala de ruido
    multi_dim_signal = np.column_stack([
        signal,
        signal * 0.7 + 0.2 * np.random.randn(points)
    ])
    
    # 2. Configurar y probar diferentes detectores
    
    # 2.1 Detector Estadístico
    print("\nProbando detector de régimen estadístico...")
    stat_detector = StatisticalRegimeDetector("statistical_detector")
    
    # Personalizar los límites de régimen para nuestros datos
    stat_detector.add_regime("low_activity", {"max_mean": 0.2, "max_std": 0.3})
    stat_detector.add_regime("normal", {
        "min_mean": -0.2, "max_mean": 0.2, 
        "min_std": 0.3, "max_std": 0.9
    })
    stat_detector.add_regime("high_activity", {"min_mean": -0.2, "min_std": 0.9})
    
    # Procesar datos por ventanas
    window_size = 30
    statistical_regimes = []
    
    for i in range(0, points, window_size):
        end = min(i + window_size, points)
        window_data = signal[i:end]
        regime = stat_detector.update(window_data)
        statistical_regimes.extend([regime] * len(window_data))
        
    print(f"Regímenes detectados estadísticamente: {set(statistical_regimes)}")
    
    # 2.2 Detector basado en tiempo
    print("\nProbando detector de régimen basado en tiempo...")
    time_detector = TimeBasedRegimeDetector("time_based_detector")
    
    # Crear una secuencia de tiempos simulados (cada punto es un minuto)
    # Comenzando desde la hora actual
    now = datetime.now()
    simulated_times = [now + timedelta(minutes=i) for i in range(points)]
    
    # Personalizar configuración de regímenes por hora
    for hour in range(24):
        if 0 <= hour < 6:
            time_detector.set_hour_regime(hour, "night")
        elif 6 <= hour < 12:
            time_detector.set_hour_regime(hour, "morning")
        elif 12 <= hour < 18:
            time_detector.set_hour_regime(hour, "afternoon")
        else:
            time_detector.set_hour_regime(hour, "evening")
    
    # Añadir un régimen especial para el almuerzo
    for day in range(7):
        time_detector.add_special_regime(day, 13, "lunch_time")
    
    # Detectar regímenes basados en tiempo
    time_regimes = []
    for simulated_time in simulated_times:
        # El detector basado en tiempo no usa los datos directamente
        regime = time_detector.detect()
        time_regimes.append(regime)
    
    print(f"Regímenes detectados por tiempo: {set(time_regimes)}")
    
    # 2.3 Detector basado en clustering
    print("\nProbando detector de régimen basado en clustering...")
    cluster_detector = ClusteringRegimeDetector("cluster_detector", n_clusters=3)
    
    # Entrenar modelo de clustering con todos los datos
    # En un caso real, esto sería un conjunto de entrenamiento histórico
    cluster_detector.fit(multi_dim_signal)
    
    # Asignar nombres específicos a los clusters
    cluster_mapping = {
        0: "cluster_regime_A",
        1: "cluster_regime_B",
        2: "cluster_regime_C"
    }
    cluster_detector.set_regime_names(cluster_mapping)
    
    # Detectar regímenes por ventanas
    clustering_regimes = []
    
    for i in range(0, points, window_size):
        end = min(i + window_size, points)
        window_data = multi_dim_signal[i:end]
        regime = cluster_detector.update(window_data)
        clustering_regimes.extend([regime] * len(window_data))
        
    print(f"Regímenes detectados por clustering: {set(clustering_regimes)}")
    
    # 2.4 Detector Híbrido
    print("\nProbando detector de régimen híbrido...")
    hybrid_detector = HybridRegimeDetector("hybrid_detector")
    
    # Configurar pesos para favorecer detección estadística y clustering
    hybrid_detector.set_detector_weight("statistical", 1.5)
    hybrid_detector.set_detector_weight("time_based", 0.5)  # Menor peso a tiempo
    hybrid_detector.set_detector_weight("clustering", 1.2)
    
    # Registrar callback para notificación de cambios
    def regime_change_notification(new_regime, old_regime):
        print(f"¡Cambio de régimen detectado! {old_regime} -> {new_regime}")
        
    hybrid_detector.add_callback(regime_change_notification)
    
    # Detectar regímenes con el detector híbrido
    hybrid_regimes = []
    
    for i in range(0, points, window_size):
        end = min(i + window_size, points)
        window_data = multi_dim_signal[i:end]
        regime = hybrid_detector.update(window_data)
        hybrid_regimes.extend([regime] * len(window_data))
        
    print(f"Regímenes detectados por el detector híbrido: {set(hybrid_regimes)}")
    
    # 3. Visualización de resultados
    try:
        plt.figure(figsize=(15, 12))
        
        # 3.1 Gráfico de la señal original
        plt.subplot(4, 1, 1)
        plt.plot(signal, 'b-')
        plt.axvline(x=200, color='r', linestyle='--')
        plt.axvline(x=400, color='r', linestyle='--')
        plt.title("Señal con Múltiples Regímenes")
        plt.ylabel("Valor")
        plt.grid(True)
        plt.annotate("Régimen 1", xy=(100, max(signal)), ha='center')
        plt.annotate("Régimen 2", xy=(300, max(signal)), ha='center')
        plt.annotate("Régimen 3", xy=(500, max(signal)), ha='center')
        
        # 3.2 Gráfico de regímenes detectados estadísticamente
        plt.subplot(4, 1, 2)
        # Convertir regímenes a valores numéricos para visualización
        unique_stat_regimes = list(set(statistical_regimes))
        stat_regime_values = [unique_stat_regimes.index(r) for r in statistical_regimes]
        plt.plot(stat_regime_values, 'g-', drawstyle='steps-post')
        plt.title("Regímenes Detectados (Estadístico)")
        plt.yticks(range(len(unique_stat_regimes)), unique_stat_regimes)
        plt.grid(True)
        
        # 3.3 Gráfico de regímenes detectados por clustering
        plt.subplot(4, 1, 3)
        unique_cluster_regimes = list(set(clustering_regimes))
        cluster_regime_values = [unique_cluster_regimes.index(r) for r in clustering_regimes]
        plt.plot(cluster_regime_values, 'r-', drawstyle='steps-post')
        plt.title("Regímenes Detectados (Clustering)")
        plt.yticks(range(len(unique_cluster_regimes)), unique_cluster_regimes)
        plt.grid(True)
        
        # 3.4 Gráfico de regímenes detectados por el detector híbrido
        plt.subplot(4, 1, 4)
        unique_hybrid_regimes = list(set(hybrid_regimes))
        hybrid_regime_values = [unique_hybrid_regimes.index(r) for r in hybrid_regimes]
        plt.plot(hybrid_regime_values, 'm-', drawstyle='steps-post')
        plt.title("Regímenes Detectados (Híbrido)")
        plt.yticks(range(len(unique_hybrid_regimes)), unique_hybrid_regimes)
        plt.xlabel("Muestra")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("regime_detection_results.png")
        print("\nGráfico guardado como 'regime_detection_results.png'")
        
    except Exception as e:
        print(f"Error al crear visualización: {str(e)}")
    
    # 4. Demostrar integración con otros componentes
    print("\nDemostración de integración con umbrales adaptativos:")
    
    # Importar ThresholdManager y ContextualThreshold de manera condicional
    try:
        from thresholds import ThresholdManager, ContextualThreshold
        
        # Crear umbral contextual y gestor
        threshold_manager = ThresholdManager()
        contextual_threshold = ContextualThreshold("contextual_with_regimes")
        
        # Configurar umbrales específicos para cada régimen detectado
        for regime in set(hybrid_regimes):
            # Asignar diferentes umbrales según el régimen
            if "high" in regime.lower():
                threshold_value = 0.8
            elif "low" in regime.lower():
                threshold_value = 0.3
            else:
                threshold_value = 0.5
            
            contextual_threshold.add_regime(regime, threshold_value)
            print(f"Configurado umbral {threshold_value} para régimen '{regime}'")
        
        # Añadir al gestor
        threshold_manager.add_threshold(contextual_threshold, set_as_current=True)
        
        # Definir función de callback para actualizar el régimen del umbral
        def update_threshold_regime(new_regime, old_regime):
            contextual_threshold.set_regime(new_regime)
            print(f"Umbral adaptado al nuevo régimen: {new_regime} (umbral={contextual_threshold.get_threshold():.2f})")
        
        # Registrar callback en el detector de régimen
        hybrid_detector.add_callback(update_threshold_regime)
        
        # Simular un cambio de régimen
        print("\nSimulando cambios de régimen para actualizar umbrales:")
        test_regimes = ["low_activity", "high_activity", "normal"]
        for regime in test_regimes:
            hybrid_detector.set_regime(regime)
            # El callback actualizará automáticamente el umbral contextual
    
    except ImportError:
        print("Módulo de umbrales no disponible para demostración de integración")
    
    # 5. Visualización avanzada - mapa de clusters
    try:
        cluster_fig = cluster_detector.plot_clusters(figsize=(10, 8))
        if cluster_fig:
            cluster_fig.savefig("regime_clusters.png")
            print("\nMapa de clusters guardado como 'regime_clusters.png'")
    except Exception as e:
        print(f"No se pudo crear mapa de clusters: {str(e)}")
    
    print("\nDemostración de detección de regímenes completada.")
