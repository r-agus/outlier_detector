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
        Actualiza el detector con nuevos datos y determina el régimen actual.
        
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
        
        # Definir regímenes predeterminados si no se proporcionan
        self.regimes = regimes if regimes is not None else {
            "low_activity": {
                "max_mean": 0.3,
                "max_std": 0.2
            },
            "normal": {
                "min_mean": 0.3,
                "max_mean": 0.7,
                "min_std": 0.2,
                "max_std": 0.5
            },
            "high_activity": {
                "min_mean": 0.7,
                "min_std": 0.5
            }
        }
        
        # Ventana para estadísticas
        self.window_size = config.adaptation.sliding_window.get("size", 100)
        self.data_window = deque(maxlen=self.window_size)
        
        # Estabilidad de régimen (para evitar cambios muy frecuentes)
        self.min_regime_duration = 10  # segundos mínimos en un régimen antes de cambiar
        
        logger.info(f"Detector estadístico de régimen inicializado con {len(self.regimes)} regímenes")
    
    def add_regime(self, regime_name: str, limits: Dict[str, float]) -> None:
        """
        Añade un nuevo régimen con sus límites estadísticos.
        
        Args:
            regime_name: Nombre del nuevo régimen
            limits: Diccionario con límites estadísticos (min_mean, max_mean, min_std, max_std, etc.)
        """
        self.regimes[regime_name] = limits
        logger.info(f"Añadido régimen '{regime_name}' con límites {limits}")
    
    def detect(self, data: np.ndarray) -> str:
        """
        Detecta el régimen basado en estadísticas de los datos.
        
        Args:
            data: Datos recientes para analizar
            
        Returns:
            Nombre del régimen detectado
        """
        # Actualizar ventana de datos
        for point in data:
            if len(point.shape) == 0 or point.shape[0] == 1:  # Punto único
                self.data_window.append(point)
            else:
                for p in point:  # Múltiples puntos
                    self.data_window.append(p)
        
        # Si no hay suficientes datos, mantener régimen actual
        if len(self.data_window) < self.window_size // 2:
            return self.current_regime
        
        # Calcular estadísticas de la ventana
        window_array = np.array(self.data_window)
        window_mean = np.mean(window_array)
        window_std = np.std(window_array)
        
        # Verificar si hemos estado en el régimen actual por suficiente tiempo
        if time.time() - self.last_change_time < self.min_regime_duration:
            return self.current_regime
        
        # Determinar régimen basado en estadísticas
        for regime_name, limits in self.regimes.items():
            matches = True
            
            # Verificar límites de media
            if "min_mean" in limits and window_mean < limits["min_mean"]:
                matches = False
            if "max_mean" in limits and window_mean > limits["max_mean"]:
                matches = False
                
            # Verificar límites de desviación estándar
            if "min_std" in limits and window_std < limits["min_std"]:
                matches = False
            if "max_std" in limits and window_std > limits["max_std"]:
                matches = False
            
            if matches:
                return regime_name
        
        # Si ningún régimen definido coincide, mantener el actual
        return self.current_regime
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Obtiene estadísticas actuales de la ventana de datos.
        
        Returns:
            Diccionario con estadísticas (mean, std, min, max)
        """
        if not self.data_window:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        window_array = np.array(self.data_window)
        return {
            "mean": float(np.mean(window_array)),
            "std": float(np.std(window_array)),
            "min": float(np.min(window_array)),
            "max": float(np.max(window_array))
        }


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
        
        # Definir regímenes por hora del día (valores predeterminados)
        self.hour_regimes = {
            # Madrugada (00-06): baja actividad
            0: "low_activity", 1: "low_activity", 2: "low_activity", 
            3: "low_activity", 4: "low_activity", 5: "low_activity",
            
            # Mañana (06-12): actividad creciente
            6: "normal", 7: "normal", 8: "normal", 
            9: "normal", 10: "normal", 11: "normal",
            
            # Tarde (12-18): alta actividad
            12: "high_activity", 13: "high_activity", 14: "high_activity", 
            15: "high_activity", 16: "high_activity", 17: "high_activity",
            
            # Noche (18-00): actividad decreciente
            18: "normal", 19: "normal", 20: "normal", 
            21: "normal", 22: "normal", 23: "low_activity"
        }
        
        # Definir regímenes por día de la semana (0=lunes, 6=domingo)
        self.weekday_modifiers = {
            0: "normal",    # Lunes
            1: "normal",    # Martes
            2: "normal",    # Miércoles
            3: "normal",    # Jueves
            4: "normal",    # Viernes
            5: "weekend",   # Sábado
            6: "weekend"    # Domingo
        }
        
        # Combinaciones especiales de día-hora
        self.special_regimes = {
            # Formato (día_semana, hora): "régimen"
            (4, 17): "friday_evening",    # Viernes tarde
            (5, 12): "saturday_noon",     # Sábado mediodía
            (6, 12): "sunday_noon"        # Domingo mediodía
        }
        
        logger.info(f"Detector de régimen basado en tiempo inicializado")
    
    def set_hour_regime(self, hour: int, regime: str) -> None:
        """
        Establece el régimen para una hora específica.
        
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
    
    def detect(self, data: np.ndarray = None) -> str:
        """
        Detecta el régimen basado en la hora y día actuales.
        
        Args:
            data: No utilizado en este detector, incluido para compatibilidad
            
        Returns:
            Nombre del régimen detectado
        """
        # Obtener la hora y día actuales
        now = datetime.now()
        current_hour = now.hour
        current_weekday = now.weekday()  # 0=lunes, 6=domingo
        
        # Verificar regímenes especiales
        if (current_weekday, current_hour) in self.special_regimes:
            return self.special_regimes[(current_weekday, current_hour)]
        
        # Combinar información de hora y día para determinar régimen
        hour_regime = self.hour_regimes.get(current_hour, "normal")
        weekday_modifier = self.weekday_modifiers.get(current_weekday, "normal")
        
        # Si es un día especial como fin de semana, puede modificar el régimen
        if weekday_modifier == "weekend":
            if hour_regime == "high_activity":
                return "weekend_high_activity"
            elif hour_regime == "normal":
                return "weekend_normal"
            else:
                return "weekend_low_activity"
        
        return hour_regime


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
    
    def fit(self, data: np.ndarray) -> None:
        """
        Entrena el modelo de clustering con datos históricos.
        
        Args:
            data: Datos para entrenamiento
        """
        try:
            # Asegurar que los datos son 2D
            if data.ndim == 1:
                train_data = data.reshape(-1, 1)
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
        # Handle different input dimensions
        # If 3D array, we need to reshape or reduce dimensionality
        original_dim = data.ndim
        if original_dim == 3:
            # Option 1: Use mean across sequences to get 2D data
            data_2d = np.mean(data, axis=1) if data.shape[1] > 1 else data.reshape(data.shape[0], -1)
        else:
            data_2d = data

        # Actualizar ventana de datos (using 2D data)
        for point in data_2d:
            if len(point.shape) == 0 or point.shape[0] == 1:  # Punto único
                self.data_window.append(point)
            else:
                for p in point:  # Múltiples puntos
                    self.data_window.append(p)
                    
        # Actualizar contador para reentrenamiento
        self.points_since_last_fit += len(data_2d)
        
        # Si no hay suficientes datos o el modelo no está entrenado, mantener régimen actual
        if len(self.data_window) < 10 or not self.is_fitted:
            return self.current_regime
        
        # Preparar datos para predicción
        window_array = np.array(self.data_window)
        if window_array.ndim == 1:
            window_array = window_array.reshape(-1, 1)
        
        # Si es tiempo de reentrenar el modelo
        if self.points_since_last_fit >= self.refit_interval:
            self.fit(window_array)
        
        # Predecir cluster - ensure proper dimensionality
        if original_dim == 3:
            # Get the most recent point from the processed data
            last_point = data_2d[-1:] 
        else:
            last_point = window_array[-1:] 

        # Ensure shape compatibility with the centroids
        if last_point.shape[1] != self.centroids.shape[1]:
            # Ajustar dimensionalidad para que coincida con los centroides
            if self.centroids.shape[1] == 2 and last_point.shape[1] == 1:
                # Si el modelo espera 2D pero tenemos 1D, duplicar la característica
                last_point = np.column_stack([last_point, last_point])
            elif self.centroids.shape[1] == 1 and last_point.shape[1] > 1:
                # Si el modelo espera 1D pero tenemos más dimensiones, tomar solo la primera
                last_point = last_point[:, :1]
        
        cluster = self.cluster_model.predict(last_point)[0]
        
        # Convertir cluster a régimen
        regime = self.cluster_to_regime.get(cluster, f"cluster_{cluster}")
        
        return regime
    
    def plot_clusters(self, figsize=(12, 8)) -> Optional[plt.Figure]:
        """
        Visualiza los clusters y la asignación de regímenes.
        
        Args:
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib o None si no es posible visualizar
        """
        if not self.is_fitted or len(self.data_window) < 10:
            logger.warning("No se puede visualizar: modelo no entrenado o datos insuficientes")
            return None
        
        try:
            # Preparar datos
            window_array = np.array(self.data_window)
            if window_array.ndim == 1:
                window_array = window_array.reshape(-1, 1)
            
            # Si los datos son multidimensionales, usar PCA para visualización
            if window_array.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                window_array_2d = pca.fit_transform(window_array)
                centroids_2d = pca.transform(self.centroids)
            elif window_array.shape[1] == 1:
                # Para datos 1D, añadir una dimensión artificial
                window_array_2d = np.column_stack([window_array, np.zeros_like(window_array)])
                centroids_2d = np.column_stack([self.centroids, np.zeros_like(self.centroids)])
            else:
                window_array_2d = window_array
                centroids_2d = self.centroids
            
            # Predecir clusters para todos los puntos
            labels = self.cluster_model.predict(window_array)
            
            # Visualizar
            fig, ax = plt.subplots(figsize=figsize)
            
            # Graficar puntos coloreados por cluster
            for i in range(self.n_clusters):
                points = window_array_2d[labels == i]
                ax.scatter(points[:, 0], points[:, 1], label=f"{self.cluster_to_regime.get(i, f'cluster_{i}')}")
            
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
            detector_name: Nombre del detector
            weight: Peso a asignar (0.0 = desactivado)
        """
        if detector_name in self.weights:
            self.weights[detector_name] = weight
            logger.info(f"Peso del detector '{detector_name}' establecido a {weight}")
        else:
            logger.warning(f"Detector '{detector_name}' no encontrado")
    
    def detect(self, data: np.ndarray) -> str:
        """
        Detecta el régimen usando un enfoque híbrido con votación ponderada.
        
        Args:
            data: Datos recientes para analizar
            
        Returns:
            Nombre del régimen detectado
        """
        # Recopilar predicciones de cada detector
        regime_votes = {}
        
        # Obtener predicción de cada detector activo
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
            # Si no hay confianza suficiente, mantener régimen actual
            detected_regime = self.current_regime
        
        logger.debug(f"Detección híbrida - Votos: {regime_votes}, " +
                    f"Ganador: {winning_regime[0]} con confianza {confidence:.2f}")
        
        return detected_regime
    
    def fit(self, data: np.ndarray) -> None:
        """
        Entrena los detectores que requieren entrenamiento (clustering).
        
        Args:
            data: Datos para entrenamiento
        """
        self.clustering_detector.fit(data)
        logger.info("Entrenamiento de detector de clustering completado")

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
