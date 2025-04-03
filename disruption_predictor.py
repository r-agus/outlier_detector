#!/usr/bin/env python3
"""
Módulo para la predicción de disrupciones en descargas de plasma.

Este módulo implementa un sistema de predicción dual que:
1. Predice si una descarga será disruptiva (clasificación binaria)
2. Estima el tiempo hasta la disrupción para casos positivos (regresión)

El sistema se actualiza incrementalmente a medida que llegan nuevos datos.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from collections import deque

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Importar configuración del sistema
from config import config

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('disruption_predictor')

class BaseDisruptionPredictor(ABC):
    """
    Clase base abstracta para modelos de predicción de disrupciones.
    Define la interfaz común que deben implementar todos los modelos.
    """
    
    def __init__(self, name: str, config_override: Dict = None):
        """
        Inicializa el predictor de disrupciones base.
        
        Args:
            name: Nombre identificativo del modelo
            config_override: Configuración personalizada que anula la configuración predeterminada
        """
        self.name = name
        self.config = config
        
        self.model = None
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.feature_names = None
        self.callbacks = []
        
    def add_callback(self, callback: Callable) -> None:
        """
        Añade una función de callback que será llamada cuando se actualice la predicción.
        
        Args:
            callback: Función a llamar con la nueva predicción
        """
        self.callbacks.append(callback)
        
    def notify_update(self, prediction_result: Dict[str, Any]) -> None:
        """
        Notifica a todos los callbacks registrados sobre una actualización en la predicción.
        
        Args:
            prediction_result: Resultado de la predicción
        """
        for callback in self.callbacks:
            callback(prediction_result)
            
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Establece los nombres de las características para el modelo.
        
        Args:
            feature_names: Lista de nombres de características
        """
        self.feature_names = feature_names
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X_train: Datos de entrenamiento (características)
            y_train: Valores objetivo de entrenamiento
            
        Returns:
            El resultado del entrenamiento
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Datos para los que realizar predicciones
            
        Returns:
            Predicciones del modelo
        """
        pass
        
    @abstractmethod
    def update_prediction(self, new_data: np.ndarray) -> Any:
        """
        Actualiza la predicción con nuevos datos de la secuencia temporal.
        
        Args:
            new_data: Nuevos datos para actualizar la predicción
            
        Returns:
            Predicción actualizada
        """
        pass
        
    def preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocesa los datos antes de usarlos para entrenamiento o predicción.
        
        Args:
            X: Datos a preprocesar
            fit: Si es True, ajusta el escalador con estos datos
            
        Returns:
            Datos preprocesados
        """
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
        
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        logger.info(f"Guardando modelo {self.name} en {filepath}")
        # Implementación específica depende del tipo de modelo
        
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo previamente guardado desde disco.
        
        Args:
            filepath: Ruta desde donde cargar el modelo
        """
        logger.info(f"Cargando modelo {self.name} desde {filepath}")
        # Implementación específica depende del tipo de modelo


class DisruptionClassifier(BaseDisruptionPredictor):
    """
    Modelo para clasificación binaria que predice si una descarga será disruptiva.
    """
    
    def __init__(self, name: str = "disruption_classifier", config_override: Dict = None):
        """
        Inicializa el clasificador de disrupciones.
        
        Args:
            name: Nombre identificativo del modelo
            config_override: Configuración personalizada
        """
        super().__init__(name, config_override)
        self._init_model()
        self.prediction_history = deque(maxlen=100)  # Valor por defecto, ajustar según configuración
        self.confidence_threshold = 0.5  # Umbral para clasificación positiva
        
    def _init_model(self) -> None:
        """
        Inicializa el modelo de clasificación según la configuración.
        """
        # Por defecto, usar Random Forest
        logger.info(f"Inicializando clasificador de disrupciones")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
            
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Entrena el clasificador con datos de descarga etiquetados.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de disrupción (1 para disrupción, 0 para no disrupción)
            
        Returns:
            El modelo entrenado
        """
        X_processed = self.preprocess_data(X_train, fit=True)
        logger.info(f"Entrenando clasificador de disrupciones con {X_processed.shape[0]} muestras")
        self.model.fit(X_processed, y_train)
        self.is_fitted = True
        return self.model
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice si ocurrirá una disrupción basándose en los datos proporcionados.
        
        Args:
            X: Datos para predecir
            
        Returns:
            Array con predicciones binarias (1=disruptivo, 0=no disruptivo)
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
            
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la probabilidad de disrupción.
        
        Args:
            X: Datos para predecir
            
        Returns:
            Array con probabilidades de disrupción
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
            
        X_processed = self.preprocess_data(X)
        return self.model.predict_proba(X_processed)[:, 1]  # Probabilidad de la clase positiva
            
    def update_prediction(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        Actualiza la predicción con nuevos datos de la secuencia temporal.
        
        Args:
            new_data: Nueva muestra de datos de la descarga
            
        Returns:
            Diccionario con la predicción actualizada y metadatos
        """
        # Preprocesar los nuevos datos
        new_data_processed = self.preprocess_data(new_data.reshape(1, -1))
        
        # Actualizar la predicción
        prob = self.model.predict_proba(new_data_processed)[0, 1]
        prediction = 1 if prob > self.confidence_threshold else 0
        
        # Actualizar el historial
        self.prediction_history.append(new_data_processed[0])
        
        # Crear resultado de predicción
        result = {
            "timestamp": datetime.now().timestamp(),
            "prediction": prediction,
            "probability": prob,
            "confidence_threshold": self.confidence_threshold,
            "is_disruptive": bool(prediction),
            "model_name": self.name,
            "features_used": self.feature_names,
            "history_samples": len(self.prediction_history)
        }
        
        # Notificar a los callbacks sobre la actualización
        self.notify_update(result)
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo de clasificación en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'confidence_threshold': self.confidence_threshold,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo de clasificación guardado en {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo de clasificación previamente guardado.
        
        Args:
            filepath: Ruta desde donde cargar el modelo
        """
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.confidence_threshold = model_data['confidence_threshold']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Modelo de clasificación cargado desde {filepath}")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el rendimiento del clasificador.
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas reales
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
            
        X_processed = self.preprocess_data(X_test)
        y_pred = self.model.predict(X_processed)
        y_prob = self.model.predict_proba(X_processed)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': 2 * (precision_score(y_test, y_pred, zero_division=0) * recall_score(y_test, y_pred, zero_division=0)) / 
                      (precision_score(y_test, y_pred, zero_division=0) + recall_score(y_test, y_pred, zero_division=0) + 1e-10)
        }
        
        return metrics


class TimeToDisruptionRegressor(BaseDisruptionPredictor):
    """
    Modelo de regresión que predice el tiempo hasta la disrupción.
    Solo hace predicciones para descargas clasificadas como disruptivas.
    """
    
    def __init__(self, name: str = "time_to_disruption_regressor", config_override: Dict = None):
        """
        Inicializa el regresor de tiempo hasta la disrupción.
        
        Args:
            name: Nombre identificativo del modelo
            config_override: Configuración personalizada
        """
        super().__init__(name, config_override)
        self._init_model()
        self.prediction_history = deque(maxlen=100)
        self.min_confidence = 0.7  # Confianza mínima para hacer una predicción de tiempo
        
    def _init_model(self) -> None:
        """
        Inicializa el modelo de regresión según la configuración.
        """
        logger.info(f"Inicializando regresor de tiempo hasta disrupción")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
            
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Entrena el regresor con datos de tiempo hasta la disrupción.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Tiempo hasta la disrupción (en segundos)
            
        Returns:
            El modelo entrenado
        """
        # Solo entrenar con datos de descargas disruptivas (tiempo > 0)
        mask = y_train > 0
        if np.sum(mask) == 0:
            logger.warning("No hay muestras disruptivas para entrenar el regresor de tiempo")
            return None
            
        X_disruptive = X_train[mask]
        y_disruptive = y_train[mask]
        
        X_processed = self.preprocess_data(X_disruptive, fit=True)
        logger.info(f"Entrenando regresor de tiempo con {X_processed.shape[0]} muestras disruptivas")
        self.model.fit(X_processed, y_disruptive)
        self.is_fitted = True
        return self.model
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice el tiempo hasta la disrupción.
        
        Args:
            X: Datos para predecir
            
        Returns:
            Array con predicciones de tiempo hasta disrupción (en segundos)
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
            
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
            
    def update_prediction(self, new_data: np.ndarray, is_disruptive: bool = None, disruptive_probability: float = None) -> Dict[str, Any]:
        """
        Actualiza la predicción de tiempo con nuevos datos.
        
        Args:
            new_data: Nueva muestra de datos de la descarga
            is_disruptive: Si la descarga ha sido clasificada como disruptiva
            disruptive_probability: Probabilidad de disrupción (del clasificador)
            
        Returns:
            Diccionario con la predicción actualizada y metadatos
        """
        # Si no se especifica si es disruptivo, no hacer predicción de tiempo
        if is_disruptive is None or not is_disruptive:
            return {
                "timestamp": datetime.now().timestamp(),
                "time_prediction": None,
                "is_disruptive": False,
                "confidence": 0.0,
                "model_name": self.name
            }
            
        # Preprocesar los nuevos datos
        new_data_processed = self.preprocess_data(new_data.reshape(1, -1))
        
        # Actualizar el historial
        self.prediction_history.append(new_data_processed[0])
        
        # Solo hacer predicción si la confianza es suficiente
        if disruptive_probability is not None and disruptive_probability < self.min_confidence:
            time_prediction = None
            confidence = disruptive_probability
        else:
            time_prediction = float(self.model.predict(new_data_processed)[0])
            
            # Obtener la confianza de la predicción (específico para RandomForest)
            confidence = disruptive_probability
            if hasattr(self.model, 'estimators_'):
                # Calcular la varianza entre los árboles como medida de confianza
                predictions = np.array([tree.predict(new_data_processed)[0] for tree in self.model.estimators_])
                std_dev = np.std(predictions)
                # Normalizar la confianza (menor desviación = mayor confianza)
                confidence = 1.0 / (1.0 + std_dev)
            
        # Crear resultado
        result = {
            "timestamp": datetime.now().timestamp(),
            "time_prediction": time_prediction,
            "time_unit": "seconds",
            "is_disruptive": bool(is_disruptive),
            "confidence": confidence,
            "model_name": self.name,
            "features_used": self.feature_names,
            "history_samples": len(self.prediction_history)
        }
        
        # Notificar a los callbacks
        self.notify_update(result)
        
        return result
        
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo de regresión en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'min_confidence': self.min_confidence,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo de regresión guardado en {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo de regresión previamente guardado.
        
        Args:
            filepath: Ruta desde donde cargar el modelo
        """
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.min_confidence = model_data['min_confidence']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Modelo de regresión cargado desde {filepath}")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el rendimiento del regresor.
        
        Args:
            X_test: Datos de prueba
            y_test: Tiempos reales hasta disrupción
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama a fit() primero.")
            
        # Solo evaluar con datos disruptivos
        mask = y_test > 0
        if np.sum(mask) == 0:
            return {
                'mae': float('nan'),
                'rmse': float('nan'),
                'r2': float('nan')
            }
            
        X_disruptive = X_test[mask]
        y_disruptive = y_test[mask]
        
        X_processed = self.preprocess_data(X_disruptive)
        y_pred = self.model.predict(X_processed)
        
        metrics = {
            'mae': mean_absolute_error(y_disruptive, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_disruptive, y_pred)),
            'r2': self.model.score(X_processed, y_disruptive)
        }
        
        return metrics


class DisruptionPredictionSystem:
    """
    Sistema orquestador que coordina el clasificador y el regresor para
    proporcionar predicciones completas sobre disrupciones.
    """
    
    def __init__(self, 
                classifier: DisruptionClassifier = None,
                regressor: TimeToDisruptionRegressor = None,
                config_override: Dict = None):
        """
        Inicializa el sistema de predicción de disrupciones.
        
        Args:
            classifier: Clasificador de disrupciones
            regressor: Regresor de tiempo hasta la disrupción
            config_override: Configuración personalizada
        """
        self.config = config
        if config_override:
            # Actualizar configuración con valores personalizados
            pass
            
        # Inicializar componentes si no se proporcionan
        self.classifier = classifier or DisruptionClassifier()
        self.regressor = regressor or TimeToDisruptionRegressor()
        
        # Historial de predicciones para seguimiento
        self.predictions_history = []
        self.max_history_size = 1000
        
        logger.info("Sistema de predicción de disrupciones inicializado")
        
    def fit(self, X_train: np.ndarray, y_disruption: np.ndarray, 
           y_time_to_disruption: np.ndarray = None) -> Dict[str, Any]:
        """
        Entrena el sistema completo con datos de entrenamiento.
        
        Args:
            X_train: Características de entrenamiento
            y_disruption: Etiquetas binarias de disrupción (1=disruptivo, 0=no disruptivo)
            y_time_to_disruption: Tiempo hasta la disrupción (0 para no disruptivas)
            
        Returns:
            Resultados del entrenamiento
        """
        # Entrenar el clasificador
        classifier_result = self.classifier.fit(X_train, y_disruption)
        
        # Entrenar el regresor (si se proporcionan datos de tiempo)
        regressor_result = None
        if y_time_to_disruption is not None:
            regressor_result = self.regressor.fit(X_train, y_time_to_disruption)
            
        return {
            "classifier_trained": self.classifier.is_fitted,
            "regressor_trained": self.regressor.is_fitted
        }
        
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Realiza una predicción completa (disrupción y tiempo).
        
        Args:
            X: Datos para los que predecir
            
        Returns:
            Diccionario con predicciones
        """
        # Primero clasificar si habrá disrupción
        disruption_proba = self.classifier.predict_proba(X)
        is_disruptive = disruption_proba > self.classifier.confidence_threshold
        
        result = {
            "timestamp": datetime.now().timestamp(),
            "is_disruptive": bool(is_disruptive[0] if isinstance(is_disruptive, np.ndarray) else is_disruptive),
            "disruption_probability": float(disruption_proba[0] if isinstance(disruption_proba, np.ndarray) else disruption_proba)
        }
        
        # Si es disruptivo, predecir tiempo hasta disrupción
        if result["is_disruptive"] and self.regressor.is_fitted:
            time_prediction = self.regressor.predict(X)
            result["time_to_disruption"] = float(time_prediction[0] if isinstance(time_prediction, np.ndarray) else time_prediction)
            result["time_unit"] = "seconds"
        else:
            result["time_to_disruption"] = None
            
        return result
        
    def update_prediction(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        Actualiza la predicción con nuevos datos de la secuencia temporal.
        
        Args:
            new_data: Nuevos datos para actualizar la predicción
            
        Returns:
            Predicción actualizada completa
        """
        # Actualizar clasificador
        classifier_result = self.classifier.update_prediction(new_data)
        
        # Actualizar regresor solo si se predice disrupción
        if classifier_result["is_disruptive"]:
            regressor_result = self.regressor.update_prediction(
                new_data, 
                is_disruptive=True,
                disruptive_probability=classifier_result["probability"]
            )
            
            # Combinar resultados
            result = {
                "timestamp": classifier_result["timestamp"],
                "is_disruptive": True,
                "disruption_probability": classifier_result["probability"],
                "time_to_disruption": regressor_result["time_prediction"],
                "time_unit": "seconds",
                "time_confidence": regressor_result["confidence"]
            }
        else:
            result = {
                "timestamp": classifier_result["timestamp"],
                "is_disruptive": False,
                "disruption_probability": classifier_result["probability"],
                "time_to_disruption": None,
                "time_unit": None,
                "time_confidence": 0.0
            }
            
        # Guardar historial
        self.predictions_history.append(result)
        if len(self.predictions_history) > self.max_history_size:
            self.predictions_history.pop(0)
            
        return result
        
    def evaluate(self, X_test: np.ndarray, y_disruption: np.ndarray, 
                y_time_to_disruption: np.ndarray = None) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del sistema completo.
        
        Args:
            X_test: Datos de prueba
            y_disruption: Etiquetas reales de disrupción
            y_time_to_disruption: Tiempos reales hasta disrupción
            
        Returns:
            Diccionario con métricas de evaluación
        """
        classifier_metrics = self.classifier.evaluate(X_test, y_disruption)
        
        regressor_metrics = {}
        if y_time_to_disruption is not None and self.regressor.is_fitted:
            regressor_metrics = self.regressor.evaluate(X_test, y_time_to_disruption)
            
        return {
            "classifier_metrics": classifier_metrics,
            "regressor_metrics": regressor_metrics
        }
        
    def save_models(self, classifier_path: str, regressor_path: str) -> None:
        """
        Guarda ambos modelos en disco.
        
        Args:
            classifier_path: Ruta para el modelo clasificador
            regressor_path: Ruta para el modelo regresor
        """
        self.classifier.save_model(classifier_path)
        self.regressor.save_model(regressor_path)
        logger.info(f"Sistema de predicción guardado en {classifier_path} y {regressor_path}")
        
    def load_models(self, classifier_path: str, regressor_path: str) -> None:
        """
        Carga ambos modelos desde disco.
        
        Args:
            classifier_path: Ruta del modelo clasificador
            regressor_path: Ruta del modelo regresor
        """
        self.classifier.load_model(classifier_path)
        self.regressor.load_model(regressor_path)
        logger.info(f"Sistema de predicción cargado desde {classifier_path} y {regressor_path}")
        
    def plot_prediction_history(self, figsize=(12, 8)) -> plt.Figure:
        """
        Visualiza el historial de predicciones.
        
        Args:
            figsize: Tamaño de la figura
            
        Returns:
            Objeto de figura de matplotlib
        """
        if not self.predictions_history:
            logger.warning("No hay historial de predicciones para visualizar")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No hay datos disponibles", ha='center', va='center')
            return fig
            
        # Extraer datos del historial
        timestamps = [p["timestamp"] for p in self.predictions_history]
        probabilities = [p["disruption_probability"] for p in self.predictions_history]
        times = []
        for p in self.predictions_history:
            if p["time_to_disruption"] is not None:
                times.append(p["time_to_disruption"])
            else:
                times.append(np.nan)
                
        # Convertir timestamps a formato de fecha
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Gráfico de probabilidad de disrupción
        ax1.plot(dates, probabilities, 'b-', label='Probabilidad de disrupción')
        ax1.axhline(y=self.classifier.confidence_threshold, color='r', linestyle='--', 
                   label=f'Umbral ({self.classifier.confidence_threshold})')
        ax1.set_ylabel('Probabilidad')
        ax1.set_title('Probabilidad de Disrupción')
        ax1.legend()
        ax1.grid(True)
        
        # Gráfico de tiempo hasta disrupción
        ax2.plot(dates, times, 'g-', label='Tiempo hasta disrupción')
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Segundos hasta disrupción')
        ax2.set_title('Predicción de Tiempo hasta Disrupción')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Crear datos sintéticos para pruebas
    np.random.seed(22)
    n_samples = 10000
    n_features = 14  # 14 señales por descarga
    
    # Crear datos de ejemplo
    X = np.random.randn(n_samples, n_features)
    # Simular etiquetas: 20% de descargas disruptivas
    y_disruption = np.random.binomial(1, 0.2, n_samples)
    # Para las disruptivas, generar tiempo hasta disrupción entre 0.1 y 10 segundos
    y_time = np.zeros(n_samples)
    disruptive_indices = np.where(y_disruption == 1)[0]
    y_time[disruptive_indices] = np.random.uniform(0.1, 10, size=len(disruptive_indices))
    
    # Dividir en entrenamiento y prueba
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_disruption_train, y_disruption_test = y_disruption[:train_size], y_disruption[train_size:]
    y_time_train, y_time_test = y_time[:train_size], y_time[train_size:]
    
    # Crear y entrenar el sistema
    system = DisruptionPredictionSystem()
    system.fit(X_train, y_disruption_train, y_time_train)
    
    # Evaluar el sistema
    metrics = system.evaluate(X_test, y_disruption_test, y_time_test)
    print("Métricas de evaluación:")
    print(f"Clasificador: {metrics['classifier_metrics']}")
    print(f"Regresor: {metrics['regressor_metrics']}")
    
    # Simular flujo de datos en tiempo real
    print("\nSimulación de predicciones en tiempo real:")
    for i in range(5):
        # Tomar una muestra del conjunto de prueba
        sample = X_test[i].reshape(1, -1)
        prediction = system.predict(sample)
        print(f"Muestra {i+1}:")
        print(f"  ¿Disruptiva? {prediction['is_disruptive']} (Prob: {prediction['disruption_probability']:.4f})")
        if prediction['time_to_disruption'] is not None:
            print(f"  Tiempo hasta disrupción: {prediction['time_to_disruption']:.2f} segundos")
        else:
            print("  Tiempo hasta disrupción: N/A")
        print()