#!/usr/bin/env python3
"""
Módulo de modelos de detección de anomalías.

Este módulo implementa varios algoritmos para detección de anomalías,
incluyendo técnicas basadas en densidad, aprendizaje no supervisado,
modelos para series temporales y enfoques de ensamble.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple, Optional, Callable
import logging
from abc import ABC, abstractmethod
from collections import deque

# Importaciones para modelos basados en densidad
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# Importaciones para modelos de aprendizaje no supervisado
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Importaciones para deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input

# Importaciones de modelos para series temporales
from hmmlearn import hmm

# Importaciones para operaciones en paralelo
from joblib import Parallel, delayed

# Importación de la configuración
from config import config

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('models')

class BaseAnomalyDetector(ABC):
    """
    Clase base abstracta para todos los detectores de anomalías.
    Define la interfaz común que deben implementar todos los modelos.
    """
    
    def __init__(self, name: str = "base_detector", config_override: Dict = None):
        """
        Inicialización del detector base.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica que sobrescribe la configuración global
        """
        self.name = name
        self.model_config = config.model
        self.config_override = config_override if config_override else {}
        self.model = None
        self.is_fitted = False
        self.threshold = None
        self.anomaly_scores_history = deque(maxlen=1000)  # Almacena puntuaciones recientes
        
        logger.info(f"Inicializando detector: {self.name}")
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto de datos es una anomalía.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        pass
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula un puntaje de anomalía para cada punto.
        Mayor valor indica mayor anomalía.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        pass
    
    def update_threshold(self, scores: np.ndarray, percentile: float = 95) -> None:
        """
        Actualiza el umbral de detección basado en puntuaciones históricas.
        
        Args:
            scores: Nuevas puntuaciones de anomalía
            percentile: Percentil para establecer el umbral
        """
        # Actualizar historia de puntuaciones
        for score in scores:
            self.anomaly_scores_history.append(score)
        
        if len(self.anomaly_scores_history) > 10:  # Asegurar datos suficientes
            # Establecer umbral basado en percentil de puntuaciones históricas
            self.threshold = np.percentile(list(self.anomaly_scores_history), percentile)
            logger.debug(f"Umbral actualizado para {self.name}: {self.threshold}")
    
    def is_anomaly(self, score: float) -> bool:
        """
        Determina si una puntuación representa una anomalía.
        
        Args:
            score: Puntuación de anomalía
            
        Returns:
            True si es anomalía, False en caso contrario
        """
        if self.threshold is None:
            return False  # Sin umbral establecido, asumimos normal
        return score > self.threshold


class LOFDetector(BaseAnomalyDetector):
    """
    Detector de anomalías basado en Local Outlier Factor (LOF).
    Detecta anomalías comparando la densidad local de un punto con sus vecinos.
    """
    
    def __init__(self, name: str = "lof_detector", config_override: Dict = None):
        """
        Inicializa el detector LOF.
        
        Args:
            name: Nombre del detector
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para LOF
        lof_config = self.model_config.lof.copy()
        if self.config_override:
            lof_config.update(self.config_override)
        
        # Inicializar modelo LOF
        self.model = LocalOutlierFactor(
            n_neighbors=lof_config["n_neighbors"],
            contamination=lof_config["contamination"],
            novelty=True,  # Habilitar predict/decision_function
        )
        
        logger.info(f"LOF inicializado con {lof_config['n_neighbors']} vecinos y contaminación {lof_config['contamination']}")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo LOF con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            self.model.fit(X)
            self.is_fitted = True
            
            # Calcular puntuaciones iniciales para establecer umbral
            scores = -self.decision_function(X)  # Negativo porque LOF devuelve scores donde valores más negativos son más anómalos
            self.update_threshold(scores)
            
            logger.info(f"LOF entrenado con {X.shape[0]} muestras")
        except Exception as e:
            logger.error(f"Error al entrenar LOF: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto de datos es una anomalía.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo LOF no ha sido entrenado. Llame a fit() primero.")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula un puntaje de anomalía para cada punto.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo LOF no ha sido entrenado. Llame a fit() primero.")
        
        # Convert input to numpy array if it's not already
        X = np.asarray(X)
        
        # Handle different input shapes - reshape 3D sequence data if needed
        if X.ndim == 3:
            original_shape = X.shape
            # Reshape to 2D: (samples*sequence_length, features)
            X_reshaped = X.reshape(-1, X.shape[2])
            # Apply decision function
            scores = -self.model.decision_function(X_reshaped)
            # Average scores for each sequence
            scores = scores.reshape(original_shape[0], original_shape[1])
            return np.mean(scores, axis=1)
        else:
            # Standard 2D input handling
            return -self.model.decision_function(X)  # Negative to make higher = more anomalous


class DBSCANDetector(BaseAnomalyDetector):
    """
    Detector basado en DBSCAN para identificar anomalías como puntos que no pertenecen a ningún cluster.
    """
    
    def __init__(self, name: str = "dbscan_detector", config_override: Dict = None):
        """
        Inicializa el detector DBSCAN.
        
        Args:
            name: Nombre del detector
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para DBSCAN
        dbscan_config = self.model_config.dbscan.copy()
        if self.config_override:
            dbscan_config.update(self.config_override)
        
        # Inicializar modelo DBSCAN
        self.model = DBSCAN(
            eps=dbscan_config["eps"],
            min_samples=dbscan_config["min_samples"],
        )
        
        # Para cálculo de puntuaciones
        self.cluster_centers = None
        self.distances_to_center = None
        
        logger.info(f"DBSCAN inicializado con eps={dbscan_config['eps']} y min_samples={dbscan_config['min_samples']}")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo DBSCAN con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            self.model.fit(X)
            self.is_fitted = True
            
            # Calcular centros de clusters y distancias
            labels = self.model.labels_
            unique_labels = set(labels)
            
            # Eliminar etiqueta -1 (ruido) si existe
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            # Calcular centros de clusters
            self.cluster_centers = []
            for label in unique_labels:
                center = X[labels == label].mean(axis=0)
                self.cluster_centers.append(center)
            
            # Si no hay clusters (todos son ruido), usar la media global
            if not self.cluster_centers:
                self.cluster_centers = [X.mean(axis=0)]
            
            # Calcular distancias iniciales para establecer umbral
            scores = self.decision_function(X)
            self.update_threshold(scores)
            
            logger.info(f"DBSCAN entrenado con {X.shape[0]} muestras, identificó {len(self.cluster_centers)} clusters")
        except Exception as e:
            logger.error(f"Error al entrenar DBSCAN: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía (no pertenece a ningún cluster).
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo DBSCAN no ha sido entrenado. Llame a fit() primero.")
            
        # Calcular puntuaciones y aplicar umbral
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, -1, 1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula puntuaciones de anomalía basadas en distancia al centro de cluster más cercano.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo DBSCAN no ha sido entrenado. Llame a fit() primero.")
        
        # Para cada punto, encontrar la distancia mínima a cualquier centro de cluster
        distances = np.zeros(X.shape[0])
        for i, point in enumerate(X):
            min_dist = float('inf')
            for center in self.cluster_centers:
                dist = np.linalg.norm(point - center)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist
        
        return distances


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Detector basado en Isolation Forest.
    Aísla observaciones evaluando la profundidad a la que son aisladas en árboles aleatorios.
    """
    
    def __init__(self, name: str = "iforest_detector", config_override: Dict = None):
        """
        Inicializa el detector Isolation Forest.
        
        Args:
            name: Nombre del detector
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para Isolation Forest
        iforest_config = self.model_config.isolation_forest.copy()
        if self.config_override:
            iforest_config.update(self.config_override)
        
        # Inicializar modelo Isolation Forest
        self.model = IsolationForest(
            n_estimators=iforest_config["n_estimators"],
            max_samples=iforest_config["max_samples"],
            contamination=iforest_config["contamination"],
            random_state=42
        )
        
        logger.info(f"Isolation Forest inicializado con {iforest_config['n_estimators']} estimadores")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo Isolation Forest con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            self.model.fit(X)
            self.is_fitted = True
            
            # Calcular puntuaciones iniciales para establecer umbral
            scores = -self.model.decision_function(X)  # Negativo porque valores más negativos son más anómalos
            self.update_threshold(scores)
            
            logger.info(f"Isolation Forest entrenado con {X.shape[0]} muestras")
        except Exception as e:
            logger.error(f"Error al entrenar Isolation Forest: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo Isolation Forest no ha sido entrenado. Llame a fit() primero.")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula un puntaje de anomalía para cada punto.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo Isolation Forest no ha sido entrenado. Llame a fit() primero.")
        
        # Handle 3D arrays by reshaping to 2D
        original_shape = X.shape
        if X.ndim == 3:
            # Reshape to 2D: (samples*sequence_length, features)
            X_reshaped = X.reshape(-1, X.shape[2])
            # Get scores and reshape back if needed
            scores = -self.model.decision_function(X_reshaped)
            if len(scores) == original_shape[0] * original_shape[1]:
                # Reshape to original first two dimensions and average across sequence
                scores = scores.reshape(original_shape[0], original_shape[1])
                return np.mean(scores, axis=1)
            return scores
        
        return -self.model.decision_function(X)  # Negativo para que mayor = más anómalo


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    Detector basado en One-Class SVM.
    Delimita una región de alta densidad en el espacio de características.
    """
    
    def __init__(self, name: str = "ocsvm_detector", config_override: Dict = None):
        """
        Inicializa el detector One-Class SVM.
        
        Args:
            name: Nombre del detector
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para One-Class SVM
        ocsvm_config = self.model_config.one_class_svm.copy()
        if self.config_override:
            ocsvm_config.update(self.config_override)
        
        # Inicializar modelo One-Class SVM
        self.model = OneClassSVM(
            kernel=ocsvm_config["kernel"],
            nu=ocsvm_config["nu"],
            gamma=ocsvm_config["gamma"]
        )
        
        logger.info(f"One-Class SVM inicializado con kernel={ocsvm_config['kernel']}, nu={ocsvm_config['nu']}")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo One-Class SVM con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            self.model.fit(X)
            self.is_fitted = True
            
            # Calcular puntuaciones iniciales para establecer umbral
            scores = -self.model.decision_function(X)  # Negativo porque valores más negativos son más anómalos
            self.update_threshold(scores)
            
            logger.info(f"One-Class SVM entrenado con {X.shape[0]} muestras")
        except Exception as e:
            logger.error(f"Error al entrenar One-Class SVM: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo One-Class SVM no ha sido entrenado. Llame a fit() primero.")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula un puntaje de anomalía para cada punto.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo One-Class SVM no ha sido entrenado. Llame a fit() primero.")
        
        # Handle 3D arrays by reshaping to 2D
        original_shape = X.shape
        if X.ndim == 3:
            # Reshape to 2D: (samples*sequence_length, features)
            X_reshaped = X.reshape(-1, X.shape[2])
            # Get scores and reshape back if needed
            scores = -self.model.decision_function(X_reshaped)
            if len(scores) == original_shape[0] * original_shape[1]:
                # Reshape to original first two dimensions and average across sequence
                scores = scores.reshape(original_shape[0], original_shape[1])
                return np.mean(scores, axis=1)
            return scores
            
        return -self.model.decision_function(X)  # Negativo para que mayor = más anómalo


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Detector basado en Autoencoder profundo.
    Identifica anomalías mediante errores de reconstrucción.
    """
    
    def __init__(self, name: str = "autoencoder_detector", config_override: Dict = None):
        """
        Inicializa el detector Autoencoder.
        
        Args:
            name: Nombre del detector
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para Autoencoder
        ae_config = self.model_config.autoencoder.copy()
        if self.config_override:
            ae_config.update(self.config_override)
        
        self.ae_config = ae_config
        self.model = None  # Se construirá en fit() cuando conozcamos la dimensión de entrada
        self.input_dim = None
        
        logger.info(f"Autoencoder inicializado con arquitectura {ae_config['layers']}")
    
    def _build_model(self, input_dim: int) -> None:
        """
        Construye la arquitectura del autoencoder.
        
        Args:
            input_dim: Dimensionalidad de los datos de entrada
        """
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for layer_size in self.ae_config["layers"][:-1]:  # Todas las capas excepto la última
            encoded = Dense(layer_size, activation=self.ae_config["activation"])(encoded)
        
        # Capa de codificación (botella de cuello)
        bottleneck_size = self.ae_config["layers"][len(self.ae_config["layers"])//2]
        bottleneck = Dense(bottleneck_size, activation=self.ae_config["activation"])(encoded)
        
        # Decoder
        decoded = bottleneck
        for layer_size in reversed(self.ae_config["layers"][:-1]):  # En orden inverso
            decoded = Dense(layer_size, activation=self.ae_config["activation"])(decoded)
        
        # Capa de salida
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Crear modelo
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mse')
        
        logger.debug(f"Modelo Autoencoder construido con dimensión de entrada {input_dim}")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el autoencoder con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            # Determinar la dimensión de entrada y construir el modelo
            self.input_dim = X.shape[1]
            self._build_model(self.input_dim)
            
            # Implementar early stopping para evitar overfitting
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Ajustar el modelo
            self.model.fit(
                X, X,  # El autoencoder intenta reconstruir la entrada
                epochs=self.ae_config["epochs"],
                batch_size=self.ae_config["batch_size"],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.is_fitted = True
            
            # Calcular errores de reconstrucción iniciales para establecer umbral
            reconstructions = self.model.predict(X)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            self.update_threshold(mse)
            
            logger.info(f"Autoencoder entrenado con {X.shape[0]} muestras durante {early_stopping.stopped_epoch} épocas")
        except Exception as e:
            logger.error(f"Error al entrenar Autoencoder: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía basado en error de reconstrucción.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo Autoencoder no ha sido entrenado. Llame a fit() primero.")
        
        # Calcular puntuaciones y aplicar umbral
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, -1, 1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula errores de reconstrucción como puntuaciones de anomalía.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con errores de reconstrucción
        """
        if not self.is_fitted:
            raise ValueError("El modelo Autoencoder no ha sido entrenado. Llame a fit() primero.")
        
        reconstructions = self.model.predict(X)
        return np.mean(np.power(X - reconstructions, 2), axis=1)


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    Detector basado en LSTM Autoencoder para datos de series temporales.
    """
    
    def __init__(self, name: str = "lstm_ae_detector", config_override: Dict = None, 
                 sequence_length: int = 10):
        """
        Inicializa el detector LSTM Autoencoder.
        
        Args:
            name: Nombre del detector
            config_override: Parámetros específicos para sobrescribir configuración global
            sequence_length: Longitud de secuencia para datos de series temporales
        """
        super().__init__(name, config_override)
        
        # Configuración específica para LSTM Autoencoder
        # Usamos la configuración del autoencoder como base
        lstm_config = self.model_config.autoencoder.copy()
        if self.config_override:
            lstm_config.update(self.config_override)
        
        self.lstm_config = lstm_config
        self.sequence_length = sequence_length
        self.model = None  # Se construirá en fit() cuando conozcamos la dimensión de entrada
        self.feature_dim = None
        
        logger.info(f"LSTM Autoencoder inicializado con secuencia de longitud {sequence_length}")
    
    def _build_model(self, feature_dim: int) -> None:
        """
        Construye la arquitectura del LSTM autoencoder.
        
        Args:
            feature_dim: Dimensión de características de cada punto en la secuencia
        """
        # Modelo secuencial
        self.model = Sequential([
            # Encoder
            LSTM(units=64, activation='tanh', input_shape=(self.sequence_length, feature_dim)),
            
            # Representación del cuello de botella
            RepeatVector(self.sequence_length),
            
            # Decoder
            LSTM(units=64, activation='tanh', return_sequences=True),
            
            # Capa de salida
            TimeDistributed(Dense(feature_dim))
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        logger.debug(f"Modelo LSTM Autoencoder construido con dimensión de características {feature_dim}")
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Convierte datos en secuencias para procesamiento LSTM.
        
        Args:
            data: Datos de entrada
            
        Returns:
            Array de secuencias
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i+self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el LSTM autoencoder con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            # Crear secuencias
            sequences = self._create_sequences(X)
            if len(sequences) == 0:
                raise ValueError(f"No se pudieron crear secuencias. La longitud de datos ({len(X)}) "
                                f"debe ser mayor que la longitud de secuencia ({self.sequence_length}).")
            
            # Determinar la dimensión de características
            self.feature_dim = X.shape[1]
            self._build_model(self.feature_dim)
            
            # Implementar early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Ajustar el modelo
            self.model.fit(
                sequences, sequences,  # El autoencoder intenta reconstruir las secuencias
                epochs=self.lstm_config["epochs"],
                batch_size=self.lstm_config["batch_size"],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.is_fitted = True
            
            # Calcular errores de reconstrucción iniciales para establecer umbral
            reconstructions = self.model.predict(sequences)
            mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
            self.update_threshold(mse)
            
            logger.info(f"LSTM Autoencoder entrenado con {len(sequences)} secuencias")
        except Exception as e:
            logger.error(f"Error al entrenar LSTM Autoencoder: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía basado en error de reconstrucción.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo LSTM Autoencoder no ha sido entrenado. Llame a fit() primero.")
        
        # Convertir a secuencias
        if len(X) < self.sequence_length:
            # Si hay pocos datos, rellenamos con los últimos puntos conocidos
            padding = self.sequence_length - len(X)
            scores = np.zeros(len(X))
            return np.ones_like(scores)  # Asumimos normal si hay pocos datos
        
        # Calcular puntuaciones
        sequences = self._create_sequences(X)
        sequence_scores = self.decision_function(sequences)
        
        # Mapear puntuaciones de secuencias de vuelta a puntos individuales
        # (tomando la media de todas las secuencias que contienen cada punto)
        scores = np.zeros(len(X))
        counts = np.zeros(len(X))
        
        for i, score in enumerate(sequence_scores):
            for j in range(self.sequence_length):
                scores[i + j] += score
                counts[i + j] += 1
        
        # Normalizar por el número de secuencias que contienen cada punto
        scores = scores / np.maximum(counts, 1)
        
        # Aplicar umbral
        return np.where(scores > self.threshold, -1, 1)
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula errores de reconstrucción como puntuaciones de anomalía para secuencias.
        
        Args:
            X: Secuencias a evaluar
            
        Returns:
            Array con errores de reconstrucción para cada secuencia
        """
        if not self.is_fitted:
            raise ValueError("El modelo LSTM Autoencoder no ha sido entrenado. Llame a fit() primero.")
        
        # Si X ya son secuencias, usar directamente
        if X.ndim == 3 and X.shape[1] == self.sequence_length:
            sequences = X
        else:
            # Convertir a secuencias si es necesario
            sequences = self._create_sequences(X)
        
        if len(sequences) == 0:
            return np.array([])  # No hay suficientes datos para formar secuencias
        
        reconstructions = self.model.predict(sequences)
        return np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))


class HMMDetector(BaseAnomalyDetector):
    """
    Detector basado en Hidden Markov Models (HMM).
    Detecta anomalías como secuencias con baja probabilidad según el modelo.
    """
    
    def __init__(self, name: str = "hmm_detector", config_override: Dict = None,
                n_components: int = 3, sequence_length: int = 10):
        """
        Inicializa el detector HMM.
        
        Args:
            name: Nombre del detector
            config_override: Configuración específica para el modelo
            n_components: Número de estados ocultos en el HMM
            sequence_length: Longitud de las secuencias para análisis
        """
        super().__init__(name, config_override)
        
        self.n_components = n_components
        self.sequence_length = sequence_length
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
        self.min_log_prob = None
        self.max_log_prob = None
        
        logger.info(f"HMM inicializado con {n_components} estados ocultos y secuencias de longitud {sequence_length}")
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Convierte datos en secuencias para procesamiento HMM.
        
        Args:
            data: Datos de entrada
            
        Returns:
            Array de secuencias
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i+self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo HMM con datos normales.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        try:
            sequences = self._create_sequences(X)
            if len(sequences) == 0:
                raise ValueError(f"No se pudieron crear secuencias. La longitud de datos ({len(X)}) "
                                f"debe ser mayor que la longitud de secuencia ({self.sequence_length}).")
            
            # Concatenar secuencias y entrenar HMM
            X_reshaped = sequences.reshape(-1, X.shape[1])
            lengths = [self.sequence_length] * len(sequences)
            
            self.model.fit(X_reshaped, lengths=lengths)
            self.is_fitted = True
            
            # Calcular puntuaciones para establecer rango y umbral
            scores = []
            for seq in sequences:
                try:
                    score = -self.model.score(seq) / self.sequence_length  # Normalizar por longitud
                    scores.append(score)
                except:
                    continue
            
            if scores:
                self.min_log_prob = min(scores)
                self.max_log_prob = max(scores)
                self.update_threshold(np.array(scores))
            
            logger.info(f"HMM entrenado con {len(sequences)} secuencias")
        except Exception as e:
            logger.error(f"Error al entrenar HMM: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía basándose en la probabilidad del HMM.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo HMM no ha sido entrenado. Llame a fit() primero.")
        
        # Manejar caso de pocos datos
        if len(X) < self.sequence_length:
            return np.ones(len(X))  # Asumir normal si hay pocos datos
        
        # Calcular puntuaciones
        sequences = self._create_sequences(X)
        sequence_scores = self.decision_function(sequences)
        
        # Mapear puntuaciones de secuencias a puntos individuales
        scores = np.zeros(len(X))
        counts = np.zeros(len(X))
        
        for i, score in enumerate(sequence_scores):
            for j in range(self.sequence_length):
                scores[i + j] += score
                counts[i + j] += 1
        
        # Normalizar por el número de secuencias que contienen cada punto
        scores = scores / np.maximum(counts, 1)
        
        # Aplicar umbral
        return np.where(scores > self.threshold, -1, 1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula puntuaciones de anomalía basadas en la probabilidad de log del HMM.
        
        Args:
            X: Secuencias a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo HMM no ha sido entrenado. Llame a fit() primero.")
        
        # Manejar formato de entrada
        if X.ndim == 3 and X.shape[1] == self.sequence_length:
            sequences = X
        else:
            sequences = self._create_sequences(X)
        
        if len(sequences) == 0:
            return np.array([])
        
        # Calcular puntuaciones para cada secuencia
        scores = []
        for seq in sequences:
            try:
                score = -self.model.score(seq) / self.sequence_length  # Normalizar por longitud
                scores.append(score)
            except:
                # En caso de error, asignar puntuación alta (probable anomalía)
                scores.append(self.max_log_prob * 1.5 if self.max_log_prob else 1000)
        
        return np.array(scores)


class EnsembleDetector(BaseAnomalyDetector):
    """
    Detector de anomalías basado en ensamble de múltiples detectores.
    Combina resultados de varios detectores para mayor robustez.
    """
    
    def __init__(self, name: str = "ensemble_detector", detectors: List[BaseAnomalyDetector] = None, 
                 weights: List[float] = None, method: str = "voting"):
        """
        Inicializa el detector de ensamble.
        
        Args:
            name: Nombre del detector
            detectors: Lista de detectores base
            weights: Pesos para cada detector (si es None, todos tienen peso igual)
            method: Método de combinación ("voting", "averaging", "max")
        """
        super().__init__(name)
        
        self.detectors = detectors if detectors else []
        self.method = method.lower()
        
        # Asignar pesos iguales si no se especifican
        if weights is None and detectors:
            self.weights = [1.0 / len(detectors)] * len(detectors)
        else:
            self.weights = weights if weights else []
        
        logger.info(f"Ensemble inicializado con {len(self.detectors)} detectores usando método '{self.method}'")
    
    def add_detector(self, detector: BaseAnomalyDetector, weight: float = None) -> None:
        """
        Añade un nuevo detector al ensamble.
        
        Args:
            detector: Detector a añadir
            weight: Peso del detector (si es None, se ajustan todos los pesos para ser iguales)
        """
        self.detectors.append(detector)
        
        # Actualizar pesos
        if weight is None:
            self.weights = [1.0 / len(self.detectors)] * len(self.detectors)
        else:
            # Normalizar los pesos existentes y añadir el nuevo
            total = sum(self.weights) + weight
            self.weights = [w / total for w in self.weights] + [weight / total]
        
        logger.debug(f"Detector {detector.name} añadido al ensamble {self.name}")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena todos los detectores del ensamble.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        if not self.detectors:
            raise ValueError("No hay detectores en el ensamble. Añada al menos un detector.")
        
        # Entrenar cada detector en paralelo si es posible
        try:
            Parallel(n_jobs=-1)(delayed(detector.fit)(X) for detector in self.detectors)
            self.is_fitted = all(detector.is_fitted for detector in self.detectors)
            
            # Calcular puntuaciones combinadas para establecer umbral
            scores = self.decision_function(X)
            self.update_threshold(scores)
            
            logger.info(f"Ensamble {self.name} entrenado con {X.shape[0]} muestras")
        except Exception as e:
            logger.error(f"Error al entrenar ensamble: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Combina las predicciones de todos los detectores.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El ensamble no ha sido entrenado. Llame a fit() primero.")
        
        # Obtener predicciones de cada detector
        all_predictions = np.array([detector.predict(X) for detector in self.detectors])
        
        if self.method == "voting":
            # Voto ponderado (cada detector vota -1 o 1 según su peso)
            weighted_votes = np.zeros((len(self.detectors), len(X)))
            for i, (pred, weight) in enumerate(zip(all_predictions, self.weights)):
                weighted_votes[i] = pred * weight
            
            # Suma de votos (positivo = normal, negativo = anomalía)
            final_votes = np.sum(weighted_votes, axis=0)
            return np.where(final_votes >= 0, 1, -1)
        
        elif self.method == "averaging" or self.method == "max":
            # Calculamos puntuaciones de cada detector
            scores = self.decision_function(X)
            # Aplicamos umbral a las puntuaciones combinadas
            return np.where(scores > self.threshold, -1, 1)
        
        else:
            raise ValueError(f"Método de ensamble '{self.method}' no reconocido")
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Combina las puntuaciones de anomalía de todos los detectores.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones combinadas
        """
        if not self.is_fitted:
            raise ValueError("El ensamble no ha sido entrenado. Llame a fit() primero.")
        
        # Ensure X is properly shaped for our base detectors
        if X.ndim == 3 and X.shape[0] == 1 and X.shape[1] == 1:
            # If X is a single sample with shape (1, 1, features)
            # Reshape to (1, features) which is what detectors expect for a single sample
            X = X.reshape(1, -1)
        
        # Obtener puntuaciones de cada detector
        all_scores = []
        for detector in self.detectors:
            try:
                scores = detector.decision_function(X)
                
                # Ensure scores has the right shape
                if hasattr(scores, 'shape'):
                    if scores.shape != (X.shape[0],) and scores.ndim == 1:
                        # If scores doesn't match the number of samples, reshape it
                        scores = np.full(X.shape[0], scores.mean())
                else:
                    # If scores is a scalar, expand to array
                    scores = np.full(X.shape[0], scores)
                
                # Normalizar puntuaciones al rango [0, 1] si es posible
                if len(detector.anomaly_scores_history) > 10:
                    min_val = min(detector.anomaly_scores_history)
                    max_val = max(detector.anomaly_scores_history)
                    range_val = max_val - min_val
                    if range_val > 0:
                        scores = (scores - min_val) / range_val
                
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Error getting scores from detector {detector.name}: {str(e)}")
                # Use a default high score (likely anomaly) in case of error
                all_scores.append(np.ones(X.shape[0]))
        
        all_scores = np.array(all_scores)
        
        if self.method == "averaging":
            # Media ponderada de puntuaciones
            return np.average(all_scores, axis=0, weights=self.weights)
        
        elif self.method == "max":
            # Máxima puntuación (más conservador, detecta más anomalías)
            return np.max(all_scores, axis=0)
        
        elif self.method == "voting":
            # Para voting, también proporcionamos una puntuación que refleja la "certeza" del voto
            weighted_votes = np.zeros((len(self.detectors), len(X)))
            for i, (scores, weight) in enumerate(zip(all_scores, self.weights)):
                # Convertir puntuaciones a decisiones binarias usando el umbral del detector
                decisions = np.where(scores > self.detectors[i].threshold, 1, 0)  # 1 = anomalía
                weighted_votes[i] = decisions * weight
            
            # Suma ponderada de votos (mayor = más probable anomalía)
            return np.sum(weighted_votes, axis=0)
        
        else:
            raise ValueError(f"Método de ensamble '{self.method}' no reconocido")


class StackingDetector(BaseAnomalyDetector):
    """
    Implementa un modelo de stacking, donde las predicciones de múltiples detectores
    se utilizan como entrada para un meta-modelo.
    """
    
    def __init__(self, name: str = "stacking_detector", 
                 base_detectors: List[BaseAnomalyDetector] = None,
                 meta_model_type: str = "isolation_forest"):
        """
        Inicializa el detector de stacking.
        
        Args:
            name: Nombre del detector
            base_detectors: Lista de detectores base
            meta_model_type: Tipo de modelo a usar como meta-modelo ("isolation_forest", "one_class_svm")
        """
        super().__init__(name)
        
        self.base_detectors = base_detectors if base_detectors else []
        self.meta_model_type = meta_model_type
        self.meta_model = None
        
        logger.info(f"Stacking inicializado con {len(self.base_detectors)} detectores base "
                   f"y meta-modelo '{meta_model_type}'")
    
    def _create_meta_model(self):
        """Crea el meta-modelo según el tipo especificado."""
        if self.meta_model_type == "isolation_forest":
            self.meta_model = IsolationForest(
                n_estimators=100, contamination=0.1, random_state=42
            )
        elif self.meta_model_type == "one_class_svm":
            self.meta_model = OneClassSVM(
                kernel="rbf", nu=0.1, gamma="scale"
            )
        else:
            raise ValueError(f"Tipo de meta-modelo '{self.meta_model_type}' no reconocido")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena los detectores base y el meta-modelo.
        
        Args:
            X: Datos de entrenamiento normalizados
        """
        if not self.base_detectors:
            raise ValueError("No hay detectores base. Añada al menos un detector.")
        
        try:
            # Entrenar detectores base
            for detector in self.base_detectors:
                detector.fit(X)
            
            # Obtener características meta (puntuaciones de los detectores base)
            meta_features = self._get_meta_features(X)
            
            # Crear y entrenar meta-modelo
            self._create_meta_model()
            self.meta_model.fit(meta_features)
            
            # Calcular puntuaciones para establecer umbral
            if hasattr(self.meta_model, "decision_function"):
                scores = -self.meta_model.decision_function(meta_features)
            elif hasattr(self.meta_model, "score_samples"):
                scores = -self.meta_model.score_samples(meta_features)
            else:
                # Si no hay función de puntuación, usar valores binarios
                scores = (self.meta_model.predict(meta_features) == -1).astype(float)
            
            self.update_threshold(scores)
            self.is_fitted = True
            
            logger.info(f"Stacking {self.name} entrenado con {X.shape[0]} muestras")
        except Exception as e:
            logger.error(f"Error al entrenar stacking: {str(e)}")
            raise
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Obtiene características meta combinando puntuaciones de detectores base.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Array con características meta
        """
        meta_features = np.zeros((X.shape[0], len(self.base_detectors)))
        
        for i, detector in enumerate(self.base_detectors):
            meta_features[:, i] = detector.decision_function(X)
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si cada punto es una anomalía utilizando el meta-modelo.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con 1 (normal) o -1 (anomalía) para cada punto
        """
        if not self.is_fitted:
            raise ValueError("El modelo stacking no ha sido entrenado. Llame a fit() primero.")
        
        # Obtener características meta y predecir con el meta-modelo
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula puntuaciones de anomalía usando el meta-modelo.
        
        Args:
            X: Datos a evaluar
            
        Returns:
            Array con puntuaciones de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo stacking no ha sido entrenado. Llame a fit() primero.")
        
        meta_features = self._get_meta_features(X)
        
        if hasattr(self.meta_model, "decision_function"):
            return -self.meta_model.decision_function(meta_features)
        elif hasattr(self.meta_model, "score_samples"):
            return -self.meta_model.score_samples(meta_features)
        else:
            # Si no hay función de puntuación, usar valores binarios
            return (self.meta_model.predict(meta_features) == -1).astype(float)


class ModelManager:
    """
    Gestiona la selección, entrenamiento, evaluación y actualización de modelos.
    """
    
    def __init__(self):
        """Inicializa el gestor de modelos."""
        self.available_models = {
            "lof": LOFDetector,
            "dbscan": DBSCANDetector,
            "isolation_forest": IsolationForestDetector,
            "one_class_svm": OneClassSVMDetector,
            "autoencoder": AutoencoderDetector,
            "lstm_autoencoder": LSTMAutoencoderDetector,
            "hmm": HMMDetector
        }
        
        self.active_detectors = {}
        self.ensemble = None
        self.stacking = None
        
        logger.info("ModelManager inicializado")
    
    def create_detector(self, detector_type: str, name: str = None, config_override: Dict = None) -> BaseAnomalyDetector:
        """
        Crea un detector del tipo especificado.
        
        Args:
            detector_type: Tipo de detector a crear
            name: Nombre del detector (opcional)
            config_override: Configuración específica para el detector
            
        Returns:
            Instancia del detector creado
        """
        if detector_type not in self.available_models:
            raise ValueError(f"Tipo de detector '{detector_type}' no reconocido")
        
        detector_class = self.available_models[detector_type]
        detector_name = name if name else f"{detector_type}_detector"
        
        detector = detector_class(name=detector_name, config_override=config_override)
        return detector
    
    def add_detector(self, detector: BaseAnomalyDetector) -> None:
        """
        Añade un detector a la lista de detectores activos.
        
        Args:
            detector: Detector a añadir
        """
        self.active_detectors[detector.name] = detector
        logger.info(f"Detector '{detector.name}' añadido a detectores activos")
    
    def create_ensemble(self, detector_names: List[str] = None, weights: List[float] = None,
                       method: str = "voting", name: str = "ensemble") -> EnsembleDetector:
        """
        Crea un detector de ensamble a partir de detectores existentes.
        
        Args:
            detector_names: Nombres de los detectores a incluir (si es None, usa todos)
            weights: Pesos para cada detector
            method: Método de combinación
            name: Nombre del ensamble
            
        Returns:
            Detector de ensamble creado
        """
        if not self.active_detectors:
            raise ValueError("No hay detectores activos para crear un ensamble")
        
        # Si no se especifican detectores, usar todos los activos
        if detector_names is None:
            detectors = list(self.active_detectors.values())
        else:
            # Verificar que todos los detectores especificados existen
            missing = [name for name in detector_names if name not in self.active_detectors]
            if missing:
                raise ValueError(f"Detectores no encontrados: {missing}")
            
            detectors = [self.active_detectors[name] for name in detector_names]
        
        self.ensemble = EnsembleDetector(name=name, detectors=detectors, weights=weights, method=method)
        logger.info(f"Ensamble '{name}' creado con {len(detectors)} detectores")
        return self.ensemble
    
    def create_stacking(self, detector_names: List[str] = None, 
                       meta_model_type: str = "isolation_forest", 
                       name: str = "stacking") -> StackingDetector:
        """
        Crea un detector de stacking a partir de detectores existentes.
        
        Args:
            detector_names: Nombres de los detectores base a incluir
            meta_model_type: Tipo de meta-modelo a usar
            name: Nombre del modelo de stacking
            
        Returns:
            Detector de stacking creado
        """
        if not self.active_detectors:
            raise ValueError("No hay detectores activos para crear un stacking")
        
        # Si no se especifican detectores, usar todos los activos
        if detector_names is None:
            base_detectors = list(self.active_detectors.values())
        else:
            # Verificar que todos los detectores especificados existen
            missing = [name for name in detector_names if name not in self.active_detectors]
            if missing:
                raise ValueError(f"Detectores no encontrados: {missing}")
            
            base_detectors = [self.active_detectors[name] for name in detector_names]
        
        self.stacking = StackingDetector(name=name, base_detectors=base_detectors, meta_model_type=meta_model_type)
        logger.info(f"Stacking '{name}' creado con {len(base_detectors)} detectores base")
        return self.stacking
    
    def evaluate_detector(self, detector: BaseAnomalyDetector, X: np.ndarray, y_true: np.ndarray = None) -> Dict[str, float]:
        """
        Evalúa el rendimiento de un detector.
        
        Args:
            detector: Detector a evaluar
            X: Datos de prueba
            y_true: Etiquetas reales (opcional, -1 para anomalías, 1 para normal)
            
        Returns:
            Diccionario con métricas de evaluación
        """
        results = {}
        
        # Obtener predicciones y puntuaciones
        try:
            y_pred = detector.predict(X)
            scores = detector.decision_function(X)
            results["score_mean"] = float(np.mean(scores))
            results["score_std"] = float(np.std(scores))
            results["anomaly_ratio"] = float(np.mean(y_pred == -1))
            
            # Si tenemos etiquetas reales, calcular métricas de clasificación
            if y_true is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                results["accuracy"] = float(accuracy_score(y_true, y_pred))
                results["precision"] = float(precision_score(y_true, y_pred, pos_label=-1))
                results["recall"] = float(recall_score(y_true, y_pred, pos_label=-1))
                results["f1"] = float(f1_score(y_true, y_pred, pos_label=-1))
                
                # ROC-AUC (no siempre aplicable)
                try:
                    # Convertir predicciones a formato binario (1 para anomalía)
                    binary_y_true = (y_true == -1).astype(int)
                    binary_scores = scores
                    results["roc_auc"] = float(roc_auc_score(binary_y_true, binary_scores))
                except:
                    results["roc_auc"] = None
            
            logger.info(f"Evaluación completada para detector '{detector.name}'")
        except Exception as e:
            logger.error(f"Error evaluando detector '{detector.name}': {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def evaluate_all_detectors(self, X: np.ndarray, y_true: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """
        Evalúa todos los detectores activos.
        
        Args:
            X: Datos de prueba
            y_true: Etiquetas reales (opcional)
            
        Returns:
            Diccionario con resultados de evaluación para cada detector
        """
        results = {}
        
        for name, detector in self.active_detectors.items():
            results[name] = self.evaluate_detector(detector, X, y_true)
        
        # Evaluar ensamble y stacking si existen
        if self.ensemble:
            results[self.ensemble.name] = self.evaluate_detector(self.ensemble, X, y_true)
        
        if self.stacking:
            results[self.stacking.name] = self.evaluate_detector(self.stacking, X, y_true)
        
        return results
    
    def incremental_update(self, detector: BaseAnomalyDetector, 
                          X_new: np.ndarray, partial_fit: bool = False) -> None:
        """
        Actualiza un detector con nuevos datos de forma incremental.
        
        Args:
            detector: Detector a actualizar
            X_new: Nuevos datos para actualización
            partial_fit: Si es True, utiliza partial_fit cuando esté disponible
        """
        if partial_fit and hasattr(detector.model, "partial_fit"):
            # Si el modelo soporta actualización incremental
            detector.model.partial_fit(X_new)
            logger.info(f"Detector '{detector.name}' actualizado incrementalmente con {len(X_new)} muestras")
        else:
            # Caso general: si tenemos acceso a los datos históricos
            if hasattr(detector, "training_data") and detector.training_data is not None:
                # Limitar tamaño del historial si es muy grande
                max_history = 10000
                if len(detector.training_data) > max_history:
                    # Mantener las muestras más recientes y algunas antiguas para evitar olvido catastrófico
                    keep_old = int(max_history * 0.3)  # 30% de datos antiguos
                    keep_recent = max_history - keep_old
                    detector.training_data = np.vstack([
                        detector.training_data[:keep_old], 
                        detector.training_data[-keep_recent:]
                    ])
                
                # Añadir nuevos datos y reentrenar
                combined_data = np.vstack([detector.training_data, X_new])
                detector.fit(combined_data)
                logger.info(f"Detector '{detector.name}' reentrenado con {len(combined_data)} muestras combinadas")
            else:
                # Si no tenemos historial, simplemente actualizamos el umbral
                scores = detector.decision_function(X_new)
                detector.update_threshold(scores)
                logger.info(f"Umbral del detector '{detector.name}' actualizado con {len(X_new)} nuevas muestras")


# Utilidades para detección de anomalías
def compare_models(data: np.ndarray, detectors: List[BaseAnomalyDetector]) -> pd.DataFrame:
    """
    Compara diferentes modelos de detección de anomalías en un conjunto de datos.
    
    Args:
        data: Datos para la comparación
        detectors: Lista de detectores a comparar
        
    Returns:
        DataFrame con resultados comparativos
    """
    results = []
    
    for detector in detectors:
        start_time = time.time()
        detector.fit(data)
        fit_time = time.time() - start_time
        
        start_time = time.time()
        predictions = detector.predict(data)
        predict_time = time.time() - start_time
        
        scores = detector.decision_function(data)
        results.append({
            "detector": detector.name,
            "fit_time": fit_time,
            "predict_time": predict_time,
            "score_mean": np.mean(scores),
            "score_std": np.std(scores),
            "anomaly_ratio": np.mean(predictions == -1)
        })
    
    return pd.DataFrame(results)

# Importaciones adicionales para visualización y evaluación
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
from typing import Dict, List, Any, Union, Tuple, Optional, Callable


# Utilidades para visualización y evaluación de modelos
def plot_anomaly_scores(scores: np.ndarray, threshold: float = None, 
                        anomalies_idx: List[int] = None, title: str = "Anomaly Scores",
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Visualiza las puntuaciones de anomalía de un detector.
    
    Args:
        scores: Puntuaciones de anomalía
        threshold: Umbral de detección (opcional)
        anomalies_idx: Índices de anomalías conocidas (opcional)
        title: Título del gráfico
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(scores, 'b-', label="Anomaly Score")
    
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.3f})")
    
    if anomalies_idx:
        ax.scatter(anomalies_idx, [scores[i] for i in anomalies_idx], 
                   color='red', marker='o', s=100, label="Known Anomalies")
    
    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_detection_results(data: np.ndarray, predictions: np.ndarray, 
                          feature_idx: Tuple[int, int] = (0, 1),
                          title: str = "Anomaly Detection Results",
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualiza los resultados de detección de anomalías en un espacio 2D.
    
    Args:
        data: Datos originales
        predictions: Predicciones (-1 para anomalías, 1 para normal)
        feature_idx: Índices de las características a visualizar
        title: Título del gráfico
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separar normales y anomalías
    normal_idx = np.where(predictions == 1)[0]
    anomaly_idx = np.where(predictions == -1)[0]
    
    # Graficar puntos normales y anómalos
    ax.scatter(data[normal_idx, feature_idx[0]], data[normal_idx, feature_idx[1]], 
              c='blue', marker='.', label="Normal")
    ax.scatter(data[anomaly_idx, feature_idx[0]], data[anomaly_idx, feature_idx[1]], 
              c='red', marker='o', s=100, label="Anomaly")
    
    ax.set_title(title)
    ax.set_xlabel(f"Feature {feature_idx[0]}")
    ax.set_ylabel(f"Feature {feature_idx[1]}")
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_feature_contribution(detector: BaseAnomalyDetector, anomaly_data: np.ndarray, 
                             normal_data: np.ndarray = None, num_features: int = 10, 
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Visualiza la contribución de cada característica a la anomalía.
    
    Args:
        detector: Detector de anomalías entrenado
        anomaly_data: Datos anómalos
        normal_data: Datos normales (para comparación)
        num_features: Número de características a mostrar
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Si el detector es un autoencoder, podemos usar error de reconstrucción por característica
    if isinstance(detector, AutoencoderDetector) and detector.is_fitted:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Obtener errores de reconstrucción
        recon_anomaly = detector.model.predict(anomaly_data)
        mse_per_feature = np.mean(np.power(anomaly_data - recon_anomaly, 2), axis=0)
        
        # Si tenemos datos normales, comparar
        if normal_data is not None:
            recon_normal = detector.model.predict(normal_data)
            normal_mse = np.mean(np.power(normal_data - recon_normal, 2), axis=0)
            
            # Ordenar por diferencia
            diff = mse_per_feature - normal_mse
            top_features = np.argsort(diff)[-num_features:]
            
            # Graficar
            features = np.arange(len(top_features))
            ax.bar(features, mse_per_feature[top_features], color='red', alpha=0.7, label="Anomaly Error")
            ax.bar(features, normal_mse[top_features], color='blue', alpha=0.7, label="Normal Error")
            ax.set_title("Feature Contribution to Anomaly (Reconstruction Error)")
            
        else:
            # Solo datos anómalos
            top_features = np.argsort(mse_per_feature)[-num_features:]
            features = np.arange(len(top_features))
            ax.bar(features, mse_per_feature[top_features], color='red', alpha=0.7)
            ax.set_title("Top Features Contributing to Anomaly")
        
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Reconstruction Error (MSE)")
        ax.set_xticks(features)
        ax.set_xticklabels(top_features)
        if normal_data is not None:
            ax.legend()
        
    else:
        # Para otros detectores, mostrar distribución de valores
        fig, ax = plt.subplots(figsize=figsize)
        
        # Seleccionar subset de características
        if anomaly_data.shape[1] > num_features:
            # Tomar las características con mayor varianza
            variances = np.var(anomaly_data, axis=0)
            selected_features = np.argsort(variances)[-num_features:]
        else:
            selected_features = np.arange(anomaly_data.shape[1])
        
        # Graficar comparación de distribuciones
        if normal_data is not None:
            data_melted = []
            
            for i, feat_idx in enumerate(selected_features):
                for val in anomaly_data[:, feat_idx]:
                    data_melted.append({"Feature": f"F{feat_idx}", "Value": val, "Type": "Anomaly"})
                    
                for val in normal_data[:, feat_idx]:
                    data_melted.append({"Feature": f"F{feat_idx}", "Value": val, "Type": "Normal"})
            
            df_melted = pd.DataFrame(data_melted)
            sns.boxplot(x="Feature", y="Value", hue="Type", data=df_melted, ax=ax)
            ax.set_title("Distribution Comparison: Normal vs Anomaly")
        else:
            # Solo datos anómalos
            df = pd.DataFrame(anomaly_data[:, selected_features], 
                             columns=[f"F{i}" for i in selected_features])
            df_melted = pd.melt(df, var_name="Feature", value_name="Value")
            sns.boxplot(x="Feature", y="Value", data=df_melted, ax=ax)
            ax.set_title("Feature Distribution in Anomalies")
    
    return fig


def hyperparameter_tuning(X: np.ndarray, detector_class: type, param_grid: Dict[str, List], 
                         cv: int = 3, scoring: str = 'neg_mean_squared_error',
                         verbose: int = 1) -> Dict[str, Any]:
    """
    Realiza búsqueda de hiperparámetros para un detector de anomalías.
    
    Args:
        X: Datos de entrenamiento
        detector_class: Clase del detector
        param_grid: Rejilla de parámetros a probar
        cv: Número de pliegues para validación cruzada
        scoring: Métrica de puntuación
        verbose: Nivel de verbosidad
        
    Returns:
        Diccionario con mejores parámetros y resultados
    """
    # Crear una instancia temporal para usar en GridSearchCV
    base_detector = detector_class()
    
    # Crear una clase wrapper para hacer compatible con scikit-learn
    class SklearnCompatibleDetector:
        def __init__(self, detector_class, params=None):
            self.detector_class = detector_class
            self.params = params or {}
            self.detector = None
        
        def fit(self, X, y=None):
            self.detector = self.detector_class(config_override=self.params)
            self.detector.fit(X)
            return self
        
        def predict(self, X):
            return self.detector.predict(X)
        
        def score(self, X, y=None):
            # Para GridSearchCV necesitamos una puntuación (mayor = mejor)
            # Usamos la inversa del error medio de reconstrucción
            scores = self.detector.decision_function(X)
            return -np.mean(scores)
        
        def get_detector(self):
            return self.detector
    
    # Crear estimador compatible con scikit-learn
    estimator = SklearnCompatibleDetector(detector_class)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid={"params": [p for p in ParameterGrid(param_grid)]},
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        return_train_score=True
    )
    
    # Ejecutar búsqueda
    grid_search.fit(X)
    
    # Obtener mejores parámetros
    best_detector = grid_search.best_estimator_.get_detector()
    
    return {
        "best_params": grid_search.best_params_["params"],
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "best_detector": best_detector
    }


def time_series_evaluation(detector: BaseAnomalyDetector, time_series_data: np.ndarray,
                         window_size: int = 100, step_size: int = 10,
                         threshold_update_freq: int = 20) -> Dict[str, List]:
    """
    Evalúa un detector en un conjunto de datos de series temporales, simulando un entorno en tiempo real.
    
    Args:
        detector: Detector de anomalías
        time_series_data: Datos de series temporales
        window_size: Tamaño de la ventana de entrenamiento/evaluación
        step_size: Tamaño del paso para mover la ventana
        threshold_update_freq: Frecuencia de actualización del umbral
        
    Returns:
        Diccionario con resultados de evaluación
    """
    results = {
        "timestamps": [],
        "scores": [],
        "predictions": [],
        "thresholds": []
    }
    
    # Entrenar con la primera ventana
    if len(time_series_data) < window_size:
        raise ValueError(f"Datos insuficientes. Se requieren al menos {window_size} puntos.")
    
    detector.fit(time_series_data[:window_size])
    current_threshold = detector.threshold
    
    # Simular procesamiento en tiempo real
    for i in range(window_size, len(time_series_data), step_size):
        window_end = min(i + step_size, len(time_series_data))
        current_data = time_series_data[i:window_end]
        
        # Obtener predicciones y puntuaciones
        scores = detector.decision_function(current_data)
        if np.isscalar(scores) or (hasattr(scores, "ndim") and scores.ndim == 0):
            scores = np.full(len(current_data), scores)
        elif len(scores) != len(current_data):
            # Si scores es un array pero no tiene la longitud correcta, lo expandimos
            scores = np.full(len(current_data), scores.mean() if len(scores) > 0 else 0)
        predictions = np.where(scores > current_threshold, -1, 1)
        
        # Almacenar resultados
        for j in range(len(current_data)):
            results["timestamps"].append(i + j)
            results["scores"].append(scores[j])
            results["predictions"].append(predictions[j])
            results["thresholds"].append(current_threshold)
        
        # Actualizar umbral periódicamente con datos recientes
        if i % (threshold_update_freq * step_size) == 0:
            recent_window = time_series_data[max(0, i - window_size):i]
            recent_scores = detector.decision_function(recent_window)
            detector.update_threshold(recent_scores)
            current_threshold = detector.threshold
    
    return results


def create_detector_suite() -> List[BaseAnomalyDetector]:
    """
    Crea una suite de detectores con configuraciones preestablecidas.
    Útil para comparación rápida o ensambles.
    
    Returns:
        Lista de detectores configurados
    """
    detectors = []
    
    # LOF con diferentes configuraciones
    detectors.append(LOFDetector(name="lof_20", config_override={"n_neighbors": 20, "contamination": 0.05}))
    detectors.append(LOFDetector(name="lof_50", config_override={"n_neighbors": 50, "contamination": 0.05}))
    
    # Isolation Forest
    detectors.append(IsolationForestDetector(name="iforest_100", 
                                           config_override={"n_estimators": 100, "contamination": 0.05}))
    
    # One-Class SVM con diferentes kernels
    detectors.append(OneClassSVMDetector(name="ocsvm_rbf", 
                                       config_override={"kernel": "rbf", "nu": 0.05}))
    
    # Autoencoder simple
    detectors.append(AutoencoderDetector(name="autoencoder",
                                       config_override={"layers": [32, 16, 8, 16, 32], "epochs": 30}))
    
    return detectors


class AnomalyDetectionResult:
    """
    Clase para almacenar y analizar resultados de detección de anomalías.
    """
    
    def __init__(self, data: np.ndarray, scores: np.ndarray, predictions: np.ndarray, 
                detector_name: str, threshold: float = None):
        """
        Inicializa el objeto de resultados.
        
        Args:
            data: Datos originales
            scores: Puntuaciones de anomalía
            predictions: Predicciones (-1 para anomalía, 1 para normal)
            detector_name: Nombre del detector
            threshold: Umbral utilizado
        """
        self.data = data
        self.scores = scores
        self.predictions = predictions
        self.detector_name = detector_name
        self.threshold = threshold
        self.anomaly_indices = np.where(predictions == -1)[0]
        self.normal_indices = np.where(predictions == 1)[0]
        
    def summary(self) -> Dict[str, Any]:
        """
        Genera un resumen de los resultados de detección.
        
        Returns:
            Diccionario con estadísticas resumidas
        """
        return {
            "detector": self.detector_name,
            "threshold": self.threshold,
            "total_samples": len(self.data),
            "anomaly_count": len(self.anomaly_indices),
            "normal_count": len(self.normal_indices),
            "anomaly_ratio": len(self.anomaly_indices) / len(self.data),
            "score_mean": np.mean(self.scores),
            "score_std": np.std(self.scores),
            "score_min": np.min(self.scores),
            "score_max": np.max(self.scores)
        }
    
    def plot_scores(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualiza las puntuaciones de anomalía.
        
        Args:
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure
        """
        return plot_anomaly_scores(
            self.scores, 
            threshold=self.threshold,
            anomalies_idx=self.anomaly_indices if len(self.anomaly_indices) < 100 else None,
            title=f"Anomaly Scores - {self.detector_name}",
            figsize=figsize
        )
    
    def plot_results_2d(self, feature_idx: Tuple[int, int] = (0, 1), 
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Visualiza los resultados en 2D.
        
        Args:
            feature_idx: Índices de características a visualizar
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure
        """
        return plot_detection_results(
            self.data,
            self.predictions,
            feature_idx=feature_idx,
            title=f"Anomaly Detection Results - {self.detector_name}",
            figsize=figsize
        )
    
    def get_top_anomalies(self, top_n: int = 10) -> np.ndarray:
        """
        Obtiene los índices de las anomalías más severas.
        
        Args:
            top_n: Número de anomalías a devolver
            
        Returns:
            Array con índices de las top anomalías
        """
        if len(self.anomaly_indices) == 0:
            return np.array([])
            
        # Ordenar anomalías por puntuación
        anomaly_scores = self.scores[self.anomaly_indices]
        top_idx = np.argsort(anomaly_scores)[-top_n:]
        
        return self.anomaly_indices[top_idx]
    
    def save_results(self, filepath: str) -> None:
        """
        Guarda los resultados en un archivo.
        
        Args:
            filepath: Ruta del archivo
        """
        results = {
            "detector": self.detector_name,
            "threshold": self.threshold,
            "scores": self.scores.tolist(),
            "predictions": self.predictions.tolist(),
            "anomaly_indices": self.anomaly_indices.tolist(),
            "summary": self.summary()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Ejemplo de uso de los detectores de anomalías
    import numpy as np
    from sklearn.datasets import make_blobs
    
    # Generar datos sintéticos
    print("Generando datos sintéticos...")
    # Datos normales (3 clusters)
    X_normal, _ = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)
    # Anomalías (puntos dispersos)
    X_anomalies = np.random.uniform(low=-15, high=15, size=(30, 10))
    # Combinar datos
    X = np.vstack([X_normal, X_anomalies])
    # Etiquetas reales (1 para normal, -1 para anomalía)
    y_true = np.ones(X.shape[0])
    y_true[-30:] = -1
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.3, random_state=42, 
        stratify=y_true  # Mantener proporción de anomalías
    )
    
    # Entrenar solo con datos normales
    X_train_normal = X_train[y_train == 1]
    
    # Crear gestor de modelos
    print("\nCreando modelos...")
    model_manager = ModelManager()
    
    # Crear y añadir detectores
    lof = model_manager.create_detector("lof", name="LOF")
    iforest = model_manager.create_detector("isolation_forest", name="IForest")
    ocsvm = model_manager.create_detector("one_class_svm", name="OCSVM")
    autoencoder = model_manager.create_detector("autoencoder", name="Autoencoder")
    
    model_manager.add_detector(lof)
    model_manager.add_detector(iforest)
    model_manager.add_detector(ocsvm)
    model_manager.add_detector(autoencoder)
    
    # Crear ensamble
    ensemble = model_manager.create_ensemble(
        detector_names=["LOF", "IForest", "OCSVM"],
        method="voting",
        name="Ensemble"
    )
    model_manager.add_detector(ensemble)
    
    # Entrenar todos los detectores
    print("\nEntrenando modelos...")
    for name, detector in model_manager.active_detectors.items():
        print(f"Entrenando {name}...")
        detector.fit(X_train_normal)
    
    # Evaluar en conjunto de prueba
    print("\nEvaluando modelos...")
    evaluation_results = model_manager.evaluate_all_detectors(X_test, y_test)
    
    # Mostrar resultados
    print("\nResultados de evaluación:")
    for name, results in evaluation_results.items():
        if "error" in results:
            print(f"{name}: Error - {results['error']}")
            continue
        
        print(f"{name}:")
        print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {results.get('precision', 'N/A'):.4f}")
        print(f"  Recall: {results.get('recall', 'N/A'):.4f}")
        print(f"  F1-Score: {results.get('f1', 'N/A'):.4f}")
        print(f"  ROC-AUC: {results.get('roc_auc', 'N/A'):.4f}")
        print(f"  Anomaly Ratio: {results['anomaly_ratio']:.4f}")
    
    # Ejemplo de detección en tiempo real (simulado)
    print("\nSimulando detección en tiempo real...")
    # Generar serie temporal sintética
    np.random.seed(42)
    time_steps = 500
    # Serie base con componente estacional
    t = np.linspace(0, 4*np.pi, time_steps)
    base_signal = np.sin(t) + 0.5 * np.sin(5*t)
    # Añadir ruido
    noisy_signal = base_signal + 0.2 * np.random.randn(time_steps)
    # Insertar anomalías
    anomaly_indices = [80, 200, 330, 420]
    for idx in anomaly_indices:
        # Anomalías como picos
        noisy_signal[idx] = base_signal[idx] + 2.5
    
    # Convertir a formato multivariado (añadir características)
    time_series_data = np.column_stack([
        noisy_signal,
        np.roll(noisy_signal, 5),  # Serie con desfase
        np.roll(noisy_signal, -5)  # Serie con adelanto
    ])
    
    # Crear detector específico para series temporales
    lstm_ae = model_manager.create_detector(
        "lstm_autoencoder", 
        name="LSTM_Autoencoder",
        config_override={"sequence_length": 10}
    )
    lstm_ae.fit(time_series_data[:100])  # Entrenar con primeros puntos
    
    # Evaluar en toda la serie
    ts_results = time_series_evaluation(
        lstm_ae,
        time_series_data,
        window_size=50,
        step_size=10
    )
    
    print(f"\nDetección de series temporales completada.")
    print(f"Anomalías detectadas: {sum(np.array(ts_results['predictions']) == -1)}")
    
    # Visualizar si es posible
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 6))
        plt.plot(ts_results["timestamps"], ts_results["scores"], 'b-', label="Anomaly Score")
        plt.plot(ts_results["timestamps"], ts_results["thresholds"], 'r--', label="Threshold")
        
        # Marcar anomalías reales
        for idx in anomaly_indices:
            if idx >= 100:  # Solo mostrar anomalías dentro del rango evaluado
                plt.axvline(x=idx, color='g', linestyle=':', alpha=0.7)
        
        plt.title("Time Series Anomaly Detection")
        plt.xlabel("Time Step")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig("time_series_results.png")
        print("\nGráfico guardado como 'time_series_results.png'")
    except Exception as e:
        print(f"No se pudo crear la visualización: {str(e)}")
    
    print("\nEjemplo completado.")
