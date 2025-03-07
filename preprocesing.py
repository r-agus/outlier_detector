#!/usr/bin/env python3
"""
Módulo de preprocesamiento de datos para el sistema de detección de anomalías.

Este módulo se encarga de la normalización, transformación y preparación de datos
antes de su análisis por los modelos de detección. Incluye técnicas de normalización,
extracción de características y procesamiento de ventanas deslizantes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union
from collections import deque
from scipy import stats
import logging
from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocessing')

class DataPreprocessor:
    """
    Clase principal para el preprocesamiento de datos.
    Maneja la normalización, extracción de características y procesamiento de ventanas.
    """
    
    def __init__(self, config_override: dict = None):
        """
        Inicializa el preprocesador con la configuración especificada.
        
        Args:
            config_override: Diccionario opcional para sobrescribir la configuración por defecto
        """
        self.config = config.preprocessing
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Ventana deslizante para estadísticas
        self.window_size = self.config.window_size
        self.data_window = deque(maxlen=self.window_size)
        
        # Estadísticas de normalización
        self.stats = {}
        self.initialized = False
        
        logger.info(f"Preprocesador inicializado con método: {self.config.normalization_method}, "
                    f"tamaño de ventana: {self.window_size}")
    
    def update_window(self, data_point: Union[np.ndarray, pd.DataFrame, pd.Series]) -> None:
        """
        Actualiza la ventana deslizante con un nuevo punto de datos.
        
        Args:
            data_point: Nuevo punto de datos a añadir a la ventana
        """
        self.data_window.append(data_point)
    
    def get_window_stats(self) -> Dict[str, np.ndarray]:
        """
        Calcula estadísticas sobre la ventana actual.
        
        Returns:
            Diccionario con estadísticas (min, max, mean, std)
        """
        if not self.data_window:
            return {}
        
        # Convertir la ventana a np.array para cálculos estadísticos
        window_array = np.array(self.data_window)
        
        stats = {
            'min': np.min(window_array, axis=0),
            'max': np.max(window_array, axis=0),
            'mean': np.mean(window_array, axis=0),
            'std': np.std(window_array, axis=0),
            'median': np.median(window_array, axis=0),
            'q1': np.percentile(window_array, 25, axis=0),
            'q3': np.percentile(window_array, 75, axis=0)
        }
        
        return stats
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Calcula y almacena estadísticas relevantes para la normalización.
        
        Args:
            data: Conjunto de datos para calcular estadísticas
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        self.stats['min'] = np.min(data, axis=0)
        self.stats['max'] = np.max(data, axis=0)
        self.stats['mean'] = np.mean(data, axis=0)
        self.stats['std'] = np.std(data, axis=0) + 1e-10  # Evitar división por cero
        
        for i in range(len(self.stats['min'])):
            # Asegurar que min != max para evitar división por cero
            if abs(self.stats['max'][i] - self.stats['min'][i]) < 1e-10:
                self.stats['max'][i] += 1e-5
        
        self.initialized = True
        logger.info("Estadísticas para normalización calculadas y almacenadas")
    
    def normalize(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Normaliza los datos según el método configurado.
        
        Args:
            data: Datos a normalizar
            
        Returns:
            Datos normalizados como np.ndarray
        """
        if not self.initialized:
            logger.warning("Normalización solicitada sin ajuste previo. Ejecutando fit() automáticamente.")
            self.fit(data)
        
        # Convertir a numpy si es necesario
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            input_data = data.values
        else:
            input_data = data
            
        # Asegurar que sea 2D para procesamiento uniforme
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        method = self.config.normalization_method.lower()
        
        if method == 'minmax':
            return (input_data - self.stats['min']) / (self.stats['max'] - self.stats['min'])
        
        elif method == 'zscore':
            return (input_data - self.stats['mean']) / self.stats['std']
        
        elif method == 'robust':
            # Normalización robusta utilizando la mediana y el rango intercuartílico
            window_stats = self.get_window_stats() if self.data_window else self.stats
            median = window_stats.get('median', self.stats['mean'])
            q1 = window_stats.get('q1', self.stats['mean'] - self.stats['std'])
            q3 = window_stats.get('q3', self.stats['mean'] + self.stats['std'])
            iqr = q3 - q1 + 1e-10  # Evitar división por cero
            return (input_data - median) / iqr
        
        else:
            logger.warning(f"Método de normalización '{method}' no reconocido. Usando 'minmax'.")
            return (input_data - self.stats['min']) / (self.stats['max'] - self.stats['min'])
    
    def extract_features(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Extrae características adicionales de los datos.
        
        Args:
            data: Datos de entrada
            
        Returns:
            Características extraídas
        """
        # Actualizar la ventana con el punto actual
        self.update_window(data)
        
        # Si la ventana no está llena aún, devolver el dato normalizado
        if len(self.data_window) < self.window_size // 2:
            return self.normalize(data)
        
        # Convertir ventana a array para cálculos
        window = np.array(self.data_window)
        
        # Extraer características simples basadas en la ventana
        if isinstance(data, np.ndarray) and data.ndim == 1:
            features = []
            norm_data = self.normalize(data)
            features.extend(norm_data)
            
            # Calcular tendencia (pendiente) de la ventana reciente
            x = np.arange(len(window))
            for dim in range(window.shape[1]):
                slope, _, _, _, _ = stats.linregress(x, window[:, dim])
                features.append(slope)
            
            # Añadir estadísticas recientes
            win_stats = self.get_window_stats()
            for stat_name in ['std', 'mean']:
                features.extend(win_stats.get(stat_name, np.zeros(window.shape[1])))
            
            return np.array(features)
        else:
            # Para matrices o DataFrames, devolver datos normalizados + estadísticas de ventana
            norm_data = self.normalize(data)
            if norm_data.ndim == 1:
                norm_data = norm_data.reshape(1, -1)
            return norm_data
    
    def process(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Procesa los datos realizando todas las etapas de preprocesamiento.
        
        Args:
            data: Datos a procesar
            
        Returns:
            Datos procesados listos para análisis
        """
        # Actualizar la ventana
        self.update_window(data)
        
        # Normalizar datos
        normalized_data = self.normalize(data)
        
        # Extraer características adicionales si es necesario
        # La implementación concreta dependerá del caso de uso específico
        
        return normalized_data
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Revierte la normalización para obtener los datos en su escala original.
        
        Args:
            normalized_data: Datos normalizados
            
        Returns:
            Datos en su escala original
        """
        if not self.initialized:
            raise ValueError("No se puede revertir la normalización sin inicialización previa con fit()")
        
        method = self.config.normalization_method.lower()
        
        if method == 'minmax':
            return normalized_data * (self.stats['max'] - self.stats['min']) + self.stats['min']
        
        elif method == 'zscore':
            return normalized_data * self.stats['std'] + self.stats['mean']
        
        elif method == 'robust':
            window_stats = self.get_window_stats() if self.data_window else self.stats
            median = window_stats.get('median', self.stats['mean'])
            q1 = window_stats.get('q1', self.stats['mean'] - self.stats['std'])
            q3 = window_stats.get('q3', self.stats['mean'] + self.stats['std'])
            iqr = q3 - q1
            return normalized_data * iqr + median
        
        else:
            return normalized_data * (self.stats['max'] - self.stats['min']) + self.stats['min']


class StreamPreprocessor(DataPreprocessor):
    """
    Extensión del preprocesador para manejar específicamente datos en tiempo real.
    Optimizado para procesamiento de stream de datos.
    """
    
    def __init__(self, config_override: dict = None):
        super().__init__(config_override)
        self.feature_buffer = {}  # Almacena características temporales
        
    def adaptive_update(self, new_data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> None:
        """
        Actualiza adaptativamente las estadísticas con nuevos datos.
        
        Args:
            new_data: Nuevos datos para actualizar estadísticas
        """
        if not self.initialized:
            self.fit(new_data)
            return
        
        # Factor de adaptación para actualización gradual
        alpha = 0.05  # Valor bajo = adaptación más lenta, valor alto = adaptación más rápida
        
        if isinstance(new_data, pd.DataFrame) or isinstance(new_data, pd.Series):
            new_data = new_data.values
            
        # Asegurar formato adecuado
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
        
        # Actualizar estadísticas con un promedio móvil exponencial
        new_min = np.min(new_data, axis=0)
        new_max = np.max(new_data, axis=0)
        new_mean = np.mean(new_data, axis=0)
        new_std = np.std(new_data, axis=0) + 1e-10
        
        self.stats['min'] = (1 - alpha) * self.stats['min'] + alpha * new_min
        self.stats['max'] = (1 - alpha) * self.stats['max'] + alpha * new_max
        self.stats['mean'] = (1 - alpha) * self.stats['mean'] + alpha * new_mean
        self.stats['std'] = (1 - alpha) * self.stats['std'] + alpha * new_std
        
        logger.debug("Estadísticas de normalización actualizadas adaptativamente")
    
    def process_stream(self, data_point: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Procesa un único punto de datos en un flujo en tiempo real.
        
        Args:
            data_point: Punto de datos a procesar
            
        Returns:
            Punto de datos procesado
        """
        # Actualizar ventana
        self.update_window(data_point)
        
        # Adaptar estadísticas si tenemos suficientes datos
        if len(self.data_window) > 10:
            window_array = np.array(self.data_window)
            self.adaptive_update(window_array)
        
        # Procesar el punto actual
        return self.process(data_point)


# Funciones de utilidad para preprocesamiento
def detect_outliers(data: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> np.ndarray:
    """
    Detecta valores atípicos en los datos.
    
    Args:
        data: Datos a analizar
        method: Método de detección ('zscore', 'iqr')
        threshold: Umbral para detección
        
    Returns:
        Máscara booleana indicando outliers
    """
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data, axis=0))
        return z_scores > threshold
    
    elif method == 'iqr':
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return (data < lower_bound) | (data > upper_bound)
    
    else:
        raise ValueError(f"Método de detección de outliers '{method}' no reconocido")

def impute_missing_values(data: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Imputa valores faltantes en los datos.
    
    Args:
        data: Datos con posibles valores faltantes (NaN)
        method: Método de imputación ('mean', 'median', 'mode', 'forward')
        
    Returns:
        Datos con valores imputados
    """
    imputed_data = data.copy()
    
    if method == 'mean':
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(imputed_data))
        imputed_data[inds] = np.take(col_mean, inds[1])
    
    elif method == 'median':
        col_median = np.nanmedian(data, axis=0)
        inds = np.where(np.isnan(imputed_data))
        imputed_data[inds] = np.take(col_median, inds[1])
    
    elif method == 'forward':
        # Forward fill (usa el último valor válido)
        for i in range(1, data.shape[0]):
            mask = np.isnan(imputed_data[i])
            imputed_data[i, mask] = imputed_data[i-1, mask]
    
    else:
        raise ValueError(f"Método de imputación '{method}' no reconocido")
    
    return imputed_data


if __name__ == "__main__":
    # Ejemplo de uso
    np.random.seed(42)
    sample_data = np.random.randn(100, 5)  # 100 puntos con 5 dimensiones
    
    # Crear instancia del preprocesador
    preprocessor = DataPreprocessor()
    
    # Ajustar y normalizar
    preprocessor.fit(sample_data)
    normalized = preprocessor.normalize(sample_data)
    
    print(f"Datos originales - Media: {np.mean(sample_data):.4f}, Std: {np.std(sample_data):.4f}")
    print(f"Datos normalizados - Media: {np.mean(normalized):.4f}, Std: {np.std(normalized):.4f}")
    
    # Probar el preprocesador de flujo
    stream_processor = StreamPreprocessor()
    stream_processor.fit(sample_data[:20])  # Ajustar con los primeros 20 puntos
    
    # Procesar un flujo simulado
    for i in range(20, 50):
        processed = stream_processor.process_stream(sample_data[i])
        print(f"Punto {i} procesado: min={np.min(processed):.4f}, max={np.max(processed):.4f}")

    # Detectar outliers
    data_with_outliers = np.random.randn(100, 5)
    data_with_outliers[::10] *= 10  # Introducir outliers
    outlier_mask = detect_outliers(data_with_outliers, method='zscore', threshold=2.5)
    print(f"Outliers detectados: {np.sum(outlier_mask)}")

    # Imputar valores faltantes
    data_with_nans = data_with_outliers.copy()
    data_with_nans[::5] = np.nan
    imputed_data = impute_missing_values(data_with_nans, method='mean')

    print(f"Valores faltantes: {np.sum(np.isnan(data_with_nans))}")
    print(f"Valores imputados: {np.sum(np.isnan(imputed_data))}")

    # Ejemplo de extracción de características
    feature_extractor = DataPreprocessor()
    feature_extractor.fit(sample_data)
    features = feature_extractor.extract_features(sample_data[0])
    print(f"Características extraídas: {features}")

    # Revertir la normalización
    original_data = preprocessor.inverse_transform(normalized)
    print(f"Datos revertidos - Media: {np.mean(original_data):.4f}, Std: {np.std(original_data):.4f}")

    # Procesar un DataFrame de Pandas
    sample_df = pd.DataFrame(sample_data, columns=[f"feature_{i}" for i in range(5)])
    processed_df = preprocessor.process(sample_df)
    print(f"Datos procesados (DataFrame): {processed_df.shape}")

    # Procesar un punto de datos con características adicionales
    sample_point = sample_data[0]
    processed_point = preprocessor.extract_features(sample_point)
    print(f"Punto procesado con características: {processed_point}")

    # Procesar un punto de datos con el preprocesador de flujo
    stream_point = sample_data[20]
    processed_stream_point = stream_processor.process_stream(stream_point)
    print(f"Punto de flujo procesado: min={np.min(processed_stream_point):.4f}, max={np.max(processed_stream_point):.4f}")

    # Actualizar adaptativamente con nuevos datos
    new_data = np.random.randn(5)
    stream_processor.adaptive_update(new_data)
    print("Estadísticas actualizadas con nuevos datos")

    # Procesar un punto de datos con el preprocesador de flujo actualizado
    processed_stream_point = stream_processor.process_stream(stream_point)
    print(f"Punto de flujo procesado: min={np.min(processed_stream_point):.4f}, max={np.max(processed_stream_point):.4f}")

    # Procesar un punto de datos con el preprocesador de flujo actualizado
    new_data = np.random.randn(5)
    stream_processor.adaptive_update(new_data)
    print("Estadísticas actualizadas con nuevos datos")

    # Procesar un punto de datos con el preprocesador de flujo actualizado
    processed_stream_point = stream_processor.process_stream(stream_point)
    print(f"Punto de flujo procesado: min={np.min(processed_stream_point):.4f}, max={np.max(processed_stream_point):.4f}")