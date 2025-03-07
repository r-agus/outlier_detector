#!/usr/bin/env python3
"""
Módulo para la implementación de umbrales adaptativos en el sistema de detección de anomalías.

Este módulo implementa diferentes estrategias de umbrales adaptativos:
- Umbrales basados en estadísticas móviles
- Umbrales probabilísticos
- Umbrales contextual-adaptativos
- Sistema de meta-umbral

Estos mecanismos permiten que el sistema se adapte automáticamente a cambios en los datos
y diferentes regímenes operativos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable
from collections import deque
from abc import ABC, abstractmethod
import logging
import time

from config import config
import matplotlib.pyplot as plt

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('thresholds')

class BaseThreshold(ABC):
    """
    Clase base abstracta para todos los mecanismos de umbral.
    Define la interfaz común que deben implementar todas las estrategias de umbral.
    """
    
    def __init__(self, name: str = "base_threshold", config_override: Dict = None):
        """
        Inicialización del umbral base.
        
        Args:
            name: Nombre del mecanismo de umbral
            config_override: Configuración específica que sobrescribe la configuración global
        """
        self.name = name
        self.threshold_config = config.threshold
        self.config_override = config_override if config_override else {}
        self.current_threshold = None
        self.history = deque(maxlen=1000)  # Historial de valores de umbral
        
        logger.info(f"Inicializando umbral adaptativo: {self.name}")
    
    def add_to_history(self, threshold_value: float) -> None:
        """
        Añade un valor de umbral al historial.
        
        Args:
            threshold_value: Valor de umbral a añadir
        """
        self.history.append((time.time(), threshold_value))
        self.current_threshold = threshold_value
    
    @abstractmethod
    def update(self, scores: np.ndarray) -> float:
        """
        Actualiza el umbral basado en nuevas puntuaciones.
        
        Args:
            scores: Nuevas puntuaciones de anomalía
            
        Returns:
            Nuevo valor de umbral
        """
        pass
    
    def get_threshold(self) -> float:
        """
        Devuelve el valor actual del umbral.
        
        Returns:
            Valor actual del umbral
        """
        if self.current_threshold is None:
            raise ValueError("El umbral no ha sido inicializado. Llame a update() primero.")
        return self.current_threshold
    
    def get_history(self) -> List[Tuple[float, float]]:
        """
        Devuelve el historial de valores de umbral.
        
        Returns:
            Lista de tuplas (timestamp, valor_umbral)
        """
        return list(self.history)


class MovingStatsThreshold(BaseThreshold):
    """
    Umbral adaptativo basado en estadísticas móviles.
    Utiliza la media y desviación estándar de las puntuaciones recientes.
    """
    
    def __init__(self, name: str = "moving_stats_threshold", config_override: Dict = None):
        """
        Inicializa el umbral de estadísticas móviles.
        
        Args:
            name: Nombre del umbral
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para este umbral
        self.config = self.threshold_config.moving_stats.copy()
        if self.config_override:
            self.config.update(self.config_override)
        
        # Ventana de puntuaciones para cálculo de estadísticas
        self.window_size = self.config.get("window_size", 100)
        self.multiplier = self.config.get("multiplier", 2.5)
        self.scores_window = deque(maxlen=self.window_size)
        
        logger.info(f"Umbral de estadísticas móviles inicializado con ventana={self.window_size}, "
                   f"multiplicador={self.multiplier}")
    
    def update(self, scores: np.ndarray) -> float:
        """
        Actualiza el umbral basado en nuevas puntuaciones usando estadísticas móviles.
        
        Args:
            scores: Nuevas puntuaciones de anomalía
            
        Returns:
            Nuevo valor de umbral
        """
        # Añadir nuevas puntuaciones a la ventana
        for score in scores:
            self.scores_window.append(score)
        
        # Si no hay suficientes datos, usar un valor predeterminado conservador
        if len(self.scores_window) < 10:
            if self.current_threshold is None:
                # Valor inicial: máximo de las primeras puntuaciones
                threshold = np.max(scores) if len(scores) > 0 else 0.5
            else:
                threshold = self.current_threshold
        else:
            # Calcular media y desviación estándar
            mean = np.mean(self.scores_window)
            std = np.std(self.scores_window)
            
            # Calcular umbral como media + multiplier * desviación estándar
            threshold = mean + self.multiplier * std
        
        # Añadir al historial y devolver
        self.add_to_history(threshold)
        logger.debug(f"Umbral actualizado a {threshold:.4f} usando estadísticas móviles")
        
        return threshold


class ProbabilisticThreshold(BaseThreshold):
    """
    Umbral adaptativo basado en percentiles de la distribución de puntuaciones.
    """
    
    def __init__(self, name: str = "probabilistic_threshold", config_override: Dict = None):
        """
        Inicializa el umbral probabilístico.
        
        Args:
            name: Nombre del umbral
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para este umbral
        self.config = self.threshold_config.probabilistic.copy()
        if self.config_override:
            self.config.update(self.config_override)
        
        # Parámetros
        self.percentile = self.config.get("percentile", 99)
        self.min_samples = self.config.get("min_samples", 30)
        self.scores_history = deque(maxlen=10000)  # Historial largo para mejor estimación
        
        logger.info(f"Umbral probabilístico inicializado con percentil={self.percentile}")
    
    def update(self, scores: np.ndarray) -> float:
        """
        Actualiza el umbral basado en el percentil de las puntuaciones históricas.
        
        Args:
            scores: Nuevas puntuaciones de anomalía
            
        Returns:
            Nuevo valor de umbral
        """
        # Añadir nuevas puntuaciones al historial
        for score in scores:
            self.scores_history.append(score)
        
        # Si no hay suficientes datos, usar un valor predeterminado
        if len(self.scores_history) < self.min_samples:
            if self.current_threshold is None:
                # Valor inicial: máximo de las primeras puntuaciones + margen
                threshold = np.max(scores) * 1.1 if len(scores) > 0 else 0.5
            else:
                threshold = self.current_threshold
        else:
            # Calcular umbral como percentil de las puntuaciones históricas
            threshold = np.percentile(list(self.scores_history), self.percentile)
        
        # Añadir al historial y devolver
        self.add_to_history(threshold)
        logger.debug(f"Umbral actualizado a {threshold:.4f} usando percentil {self.percentile}")
        
        return threshold
    
    def estimate_threshold_for_false_positive_rate(self, target_fpr: float = 0.01) -> float:
        """
        Estima el umbral que produciría una tasa de falsos positivos específica.
        
        Args:
            target_fpr: Tasa de falsos positivos objetivo (0.01 = 1%)
            
        Returns:
            Valor estimado del umbral
        """
        if len(self.scores_history) < self.min_samples:
            return self.get_threshold() if self.current_threshold is not None else 0.5
        
        # Percentil correspondiente a la tasa de falsos positivos
        percentile = 100 * (1 - target_fpr)
        return np.percentile(list(self.scores_history), percentile)


class ContextualThreshold(BaseThreshold):
    """
    Umbral adaptativo que varía según el contexto o régimen de operación.
    """
    
    def __init__(self, name: str = "contextual_threshold", config_override: Dict = None):
        """
        Inicializa el umbral contextual.
        
        Args:
            name: Nombre del umbral
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para este umbral
        self.config = self.threshold_config.contextual.copy()
        if self.config_override:
            self.config.update(self.config_override)
        
        # Umbral predeterminado y umbrales específicos por régimen
        self.default_threshold = self.config.get("default_threshold", 0.5)
        self.regime_thresholds = self.config.get("regime_thresholds", {
            "normal": 0.5,
            "peak": 0.7,
            "off_peak": 0.4
        })
        
        # Régimen actual
        self.current_regime = "normal"
        
        logger.info(f"Umbral contextual inicializado con regímenes: {list(self.regime_thresholds.keys())}")
    
    def update(self, scores: np.ndarray) -> float:
        """
        Actualiza el umbral basado en el régimen actual.
        
        Args:
            scores: Nuevas puntuaciones de anomalía (no utilizadas directamente)
            
        Returns:
            Umbral actual según el régimen
        """
        # Obtener umbral para el régimen actual
        threshold = self.regime_thresholds.get(self.current_regime, self.default_threshold)
        
        # Añadir al historial y devolver
        self.add_to_history(threshold)
        logger.debug(f"Umbral contextual: {threshold:.4f} para régimen '{self.current_regime}'")
        
        return threshold
    
    def set_regime(self, regime: str) -> None:
        """
        Establece el régimen actual de operación.
        
        Args:
            regime: Nombre del régimen
        """
        if regime not in self.regime_thresholds:
            logger.warning(f"Régimen '{regime}' no definido. Usando umbral predeterminado.")
        
        self.current_regime = regime
        logger.info(f"Régimen cambiado a '{regime}'")
        
        # Actualizar umbral inmediatamente
        self.update(np.array([]))
    
    def add_regime(self, regime_name: str, threshold_value: float) -> None:
        """
        Añade un nuevo régimen con su umbral asociado.
        
        Args:
            regime_name: Nombre del nuevo régimen
            threshold_value: Valor de umbral para el régimen
        """
        self.regime_thresholds[regime_name] = threshold_value
        logger.info(f"Añadido régimen '{regime_name}' con umbral {threshold_value}")


class MetaThreshold(BaseThreshold):
    """
    Sistema de meta-umbral que ajusta dinámicamente otros umbrales
    basado en rendimiento y retroalimentación.
    """
    
    def __init__(self, name: str = "meta_threshold", thresholds: List[BaseThreshold] = None,
                config_override: Dict = None):
        """
        Inicializa el meta-umbral.
        
        Args:
            name: Nombre del umbral
            thresholds: Lista de umbrales a gestionar
            config_override: Parámetros específicos para sobrescribir configuración global
        """
        super().__init__(name, config_override)
        
        # Configuración específica para este umbral
        self.config = self.threshold_config.meta.copy()
        if self.config_override:
            self.config.update(self.config_override)
        
        # Parámetros
        self.monitor_interval = self.config.get("monitor_interval", 60)  # segundos
        self.adjustment_factor = self.config.get("adjustment_factor", 0.1)
        
        # Umbrales gestionados
        self.thresholds = thresholds if thresholds is not None else []
        self.threshold_weights = {threshold.name: 1.0 for threshold in self.thresholds}
        
        # Métricas de rendimiento
        self.last_monitor_time = time.time()
        self.false_positives = 0
        self.false_negatives = 0
        
        logger.info(f"Meta-umbral inicializado con {len(self.thresholds)} umbrales")
    
    def add_threshold(self, threshold: BaseThreshold, weight: float = 1.0) -> None:
        """
        Añade un umbral para ser gestionado.
        
        Args:
            threshold: Umbral a añadir
            weight: Peso inicial para este umbral
        """
        self.thresholds.append(threshold)
        self.threshold_weights[threshold.name] = weight
        logger.info(f"Añadido umbral '{threshold.name}' con peso {weight}")
    
    def update(self, scores: np.ndarray) -> float:
        """
        Actualiza el umbral basado en la combinación ponderada de otros umbrales.
        
        Args:
            scores: Nuevas puntuaciones de anomalía
            
        Returns:
            Umbral combinado
        """
        # Actualizar todos los umbrales gestionados
        threshold_values = []
        
        for threshold in self.thresholds:
            threshold_value = threshold.update(scores)
            weight = self.threshold_weights.get(threshold.name, 1.0)
            threshold_values.append((threshold_value, weight))
        
        # Calcular umbral combinado (promedio ponderado)
        if threshold_values:
            threshold = sum(value * weight for value, weight in threshold_values) / sum(weight for _, weight in threshold_values)
        else:
            # Si no hay umbrales, usar valor predeterminado
            threshold = 0.5 if self.current_threshold is None else self.current_threshold
        
        # Verificar si es tiempo de monitorear y ajustar
        current_time = time.time()
        if current_time - self.last_monitor_time > self.monitor_interval:
            self._adjust_weights()
            self.last_monitor_time = current_time
        
        # Añadir al historial y devolver
        self.add_to_history(threshold)
        logger.debug(f"Meta-umbral actualizado a {threshold:.4f}")
        
        return threshold
    
    def _adjust_weights(self) -> None:
        """
        Ajusta los pesos de los umbrales basado en su rendimiento.
        Se llama periódicamente según el intervalo de monitoreo.
        """
        # Si no tenemos datos de rendimiento, no ajustar
        if self.false_positives == 0 and self.false_negatives == 0:
            return
        
        # Calcular métricas de equilibrio
        if self.false_positives > self.false_negatives:
            # Hay más falsos positivos, aumentar umbral
            for threshold in self.thresholds:
                current_weight = self.threshold_weights.get(threshold.name, 1.0)
                # Umbrales más altos reducen falsos positivos
                if any(isinstance(threshold, cls) for cls in [ProbabilisticThreshold, MovingStatsThreshold]):
                    self.threshold_weights[threshold.name] = current_weight * (1 + self.adjustment_factor)
        else:
            # Hay más falsos negativos, reducir umbral
            for threshold in self.thresholds:
                current_weight = self.threshold_weights.get(threshold.name, 1.0)
                # Umbrales más bajos reducen falsos negativos
                if any(isinstance(threshold, cls) for cls in [ProbabilisticThreshold, MovingStatsThreshold]):
                    self.threshold_weights[threshold.name] = current_weight * (1 - self.adjustment_factor)
        
        # Reiniciar contadores
        self.false_positives = 0
        self.false_negatives = 0
        
        logger.info("Pesos de umbrales ajustados basados en rendimiento")
    
    def register_feedback(self, was_anomaly: bool, predicted_anomaly: bool) -> None:
        """
        Registra retroalimentación sobre predicciones para ajuste posterior.
        
        Args:
            was_anomaly: Si realmente era una anomalía
            predicted_anomaly: Si fue predicho como anomalía
        """
        if predicted_anomaly and not was_anomaly:
            # Falso positivo
            self.false_positives += 1
        elif not predicted_anomaly and was_anomaly:
            # Falso negativo
            self.false_negatives += 1


class ThresholdManager:
    """
    Gestiona múltiples estrategias de umbral y proporciona una interfaz unificada.
    """
    
    def __init__(self):
        """
        Inicializa el gestor de umbrales.
        """
        self.thresholds = {}
        self.current_strategy = None
        self.meta_threshold = None
        
        logger.info("Gestor de umbrales inicializado")
    
    def add_threshold(self, threshold: BaseThreshold, set_as_current: bool = False) -> None:
        """
        Añade un umbral al gestor.
        
        Args:
            threshold: Umbral a añadir
            set_as_current: Si establecer como estrategia actual
        """
        self.thresholds[threshold.name] = threshold
        
        if set_as_current:
            self.current_strategy = threshold.name
        
        logger.info(f"Umbral '{threshold.name}' añadido al gestor")
    
    def set_current_strategy(self, threshold_name: str) -> None:
        """
        Establece la estrategia de umbral actual.
        
        Args:
            threshold_name: Nombre del umbral a usar
        """
        if threshold_name not in self.thresholds:
            raise ValueError(f"Umbral '{threshold_name}' no encontrado")
        
        self.current_strategy = threshold_name
        logger.info(f"Estrategia de umbral cambiada a '{threshold_name}'")
    
    def setup_meta_threshold(self, threshold_names: List[str] = None,
                           weights: List[float] = None) -> None:
        """
        Configura un meta-umbral con umbrales existentes.
        
        Args:
            threshold_names: Nombres de los umbrales a incluir
            weights: Pesos correspondientes a los umbrales
        """
        if threshold_names is None:
            threshold_names = list(self.thresholds.keys())
        
        if weights is None:
            weights = [1.0] * len(threshold_names)
        
        if len(weights) != len(threshold_names):
            raise ValueError("La cantidad de pesos debe coincidir con la cantidad de umbrales")
        
        # Crear meta-umbral
        self.meta_threshold = MetaThreshold(name="meta_threshold")
        
        # Añadir umbrales seleccionados
        for name, weight in zip(threshold_names, weights):
            if name in self.thresholds:
                self.meta_threshold.add_threshold(self.thresholds[name], weight)
            else:
                logger.warning(f"Umbral '{name}' no encontrado, ignorado en meta-umbral")
        
        # Añadir meta-umbral al gestor
        self.add_threshold(self.meta_threshold, set_as_current=True)
        
        logger.info(f"Meta-umbral configurado con {len(threshold_names)} umbrales")
    
    def get_threshold(self) -> float:
        """
        Obtiene el valor de umbral actual.
        
        Returns:
            Valor actual del umbral
        """
        if self.current_strategy is None:
            raise ValueError("No se ha establecido una estrategia de umbral")
        
        return self.thresholds[self.current_strategy].get_threshold()
    
    def update(self, scores: np.ndarray) -> float:
        """
        Actualiza el umbral actual con nuevas puntuaciones.
        
        Args:
            scores: Nuevas puntuaciones de anomalía
            
        Returns:
            Nuevo valor de umbral
        """
        if self.current_strategy is None:
            raise ValueError("No se ha establecido una estrategia de umbral")
        
        return self.thresholds[self.current_strategy].update(scores)


# Utilidades para análisis y visualización de umbrales
def plot_threshold_history(threshold: BaseThreshold, figsize=(12, 6)) -> plt.Figure:
    """
    Visualiza el historial de valores de umbral.
    
    Args:
        threshold: Objeto de umbral
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import date2num
        import datetime
        
        history = threshold.get_history()
        if not history:
            logger.warning("No hay historial para visualizar")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No hay datos de historial disponibles", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Convertir timestamps a datetime
        times = [datetime.datetime.fromtimestamp(t) for t, _ in history]
        values = [v for _, v in history]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, values, 'b-', marker='o', markersize=3)
        
        ax.set_title(f"Historial de Umbral: {threshold.name}")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Valor de Umbral")
        ax.grid(True)
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        logger.warning("matplotlib no disponible para visualización")
        return None


def compare_thresholds(scores: np.ndarray, thresholds: List[BaseThreshold]) -> pd.DataFrame:
    """
    Compara diferentes estrategias de umbral en un conjunto de puntuaciones.
    
    Args:
        scores: Puntuaciones de anomalía
        thresholds: Lista de umbrales a comparar
        
    Returns:
        DataFrame con resultados comparativos
    """
    results = []
    
    for threshold in thresholds:
        # Actualizar umbral
        threshold_value = threshold.update(scores)
        
        # Calcular métricas
        anomaly_predictions = scores > threshold_value
        anomaly_rate = np.mean(anomaly_predictions)
        
        results.append({
            'threshold_name': threshold.name,
            'threshold_value': threshold_value,
            'anomaly_rate': anomaly_rate,
            'max_score': np.max(scores),
            'mean_score': np.mean(scores),
            '95th_percentile': np.percentile(scores, 95)
        })
    
    return pd.DataFrame(results)


def adaptive_threshold_evaluation(thresholds: Dict[str, BaseThreshold], 
                               time_series_data: np.ndarray,
                               anomaly_scores_func: Callable[[np.ndarray], np.ndarray],
                               true_anomalies: np.ndarray = None,
                               window_size: int = 20) -> Dict[str, Any]:
    """
    Evalúa diferentes estrategias de umbral en una serie temporal.
    
    Args:
        thresholds: Diccionario de umbrales a evaluar
        time_series_data: Datos de series temporales
        anomaly_scores_func: Función para calcular puntuaciones de anomalía
        true_anomalies: Array booleano con verdaderos positivos (opcional)
        window_size: Tamaño de ventana para evaluación deslizante
        
    Returns:
        Diccionario con resultados de evaluación
    """
    results = {
        'timestamps': [],
        'scores': [],
        'threshold_values': {name: [] for name in thresholds.keys()},
        'predictions': {name: [] for name in thresholds.keys()},
        'metrics': {name: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for name in thresholds.keys()}
    }
    
    # Simular procesamiento en tiempo real
    for i in range(0, len(time_series_data) - window_size, window_size // 2):
        # Obtener ventana actual
        window = time_series_data[i:i+window_size]
        
        # Calcular puntuaciones de anomalía
        scores = anomaly_scores_func(window)
        
        # Actualizar cada umbral y calcular predicciones
        for name, threshold in thresholds.items():
            threshold_value = threshold.update(scores)
            predictions = scores > threshold_value
            
            # Almacenar resultados
            results['threshold_values'][name].extend([threshold_value] * len(scores))
            results['predictions'][name].extend(predictions)
            
            # Actualizar métricas si hay etiquetas verdaderas
            if true_anomalies is not None:
                window_true = true_anomalies[i:i+window_size]
                
                # Calcular métricas
                tp = np.sum(predictions & window_true)
                fp = np.sum(predictions & ~window_true)
                tn = np.sum(~predictions & ~window_true)
                fn = np.sum(~predictions & window_true)
                
                results['metrics'][name]['tp'] += tp
                results['metrics'][name]['fp'] += fp
                results['metrics'][name]['tn'] += tn
                results['metrics'][name]['fn'] += fn
        
        # Almacenar puntuaciones
        results['timestamps'].extend(range(i, i+len(scores)))
        results['scores'].extend(scores)
    
    # Calcular métricas adicionales
    if true_anomalies is not None:
        for name in thresholds.keys():
            metrics = results['metrics'][name]
            metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp']) if metrics['tp'] + metrics['fp'] > 0 else 0
            metrics['recall'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if metrics['tp'] + metrics['fn'] > 0 else 0
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if metrics['precision'] + metrics['recall'] > 0 else 0
            metrics['accuracy'] = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'])
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso de umbrales adaptativos
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generar datos sintéticos
    np.random.seed(42)
    normal_scores = np.random.normal(0, 1, 1000)  # Puntuaciones normales
    anomaly_scores = np.random.normal(4, 1, 50)   # Puntuaciones anómalas
    
    # Mezclar algunas anomalías
    all_scores = np.copy(normal_scores)
    anomaly_indices = np.random.choice(range(len(all_scores)), size=len(anomaly_scores), replace=False)
    all_scores[anomaly_indices] = anomaly_scores
    
    # Crear diferentes umbrales
    moving_stats = MovingStatsThreshold(name="MovingStats")
    probabilistic = ProbabilisticThreshold(name="Probabilistic")
    contextual = ContextualThreshold(name="Contextual")
    
    # Crear gestor de umbrales
    manager = ThresholdManager()
    manager.add_threshold(moving_stats)
    manager.add_threshold(probabilistic)
    manager.add_threshold(contextual)
    
    # Configurar meta-umbral
    manager.setup_meta_threshold(weights=[1.0, 1.5, 0.8])
    
    # Procesar datos en lotes
    batch_size = 50
    thresholds_history = {
        "MovingStats": [],
        "Probabilistic": [],
        "Contextual": [],
        "meta_threshold": []
    }
    timestamps = []
    
    print("Procesando datos en lotes...")
    for i in range(0, len(all_scores), batch_size):
        batch = all_scores[i:i+batch_size]
        batch_time = i / batch_size  # Timestamp simulado
        timestamps.append(batch_time)
        
        # Actualizar cada umbral
        for name, threshold in manager.thresholds.items():
            threshold_value = threshold.update(batch)
            thresholds_history[name].append(threshold_value)
            print(f"Lote {i//batch_size + 1}: {name} = {threshold_value:.4f}")
        
        # Simular cambio de régimen en el punto medio
        if i == len(all_scores) // 2:
            print("\nCambio de régimen detectado, ajustando umbral contextual...\n")
            contextual.set_regime("peak")
    
    # Visualizar resultados
    try:
        plt.figure(figsize=(14, 10))
        
        # Gráfico de puntuaciones y anomalías
        plt.subplot(2, 1, 1)
        plt.plot(all_scores, 'b-', alpha=0.5, label="Scores")
        plt.scatter(anomaly_indices, all_scores[anomaly_indices], color='r', marker='x', s=100, label="Anomalías")
        plt.title("Puntuaciones de anomalía")
        plt.xlabel("Muestra")
        plt.ylabel("Puntuación")
        plt.legend()
        plt.grid(True)
        
        # Gráfico de umbrales
        plt.subplot(2, 1, 2)
        batch_indices = range(0, len(all_scores), batch_size)
        for name in thresholds_history.keys():
            values = thresholds_history[name]
            if len(values) > 0:  # Si hay valores registrados
                plt.step(batch_indices[:len(values)], values, label=name, where='post')
        
        plt.title("Evolución de umbrales adaptativos")
        plt.xlabel("Muestra")
        plt.ylabel("Valor de umbral")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("adaptive_thresholds.png")
        print("\nGráfico guardado como 'adaptive_thresholds.png'")
        
    except Exception as e:
        print(f"No se pudo crear visualización: {str(e)}")
    
    # Comparar métodos de umbral
    print("\nComparación de métodos de umbral:")
    results = []
    for name, threshold in manager.thresholds.items():
        # Obtener valores finales
        threshold_value = threshold.get_threshold()
        predictions = all_scores > threshold_value
        anomalies = np.where(predictions)[0]
        
        # Calcular precisión (considerando anomaly_indices como verdaderos positivos)
        true_positives = np.intersect1d(anomalies, anomaly_indices).shape[0]
        precision = true_positives / max(len(anomalies), 1)
        
        # Calcular recall
        recall = true_positives / len(anomaly_indices)
        
        results.append({
            'method': name,
            'threshold': threshold_value,
            'anomalies_found': len(anomalies),
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / max((precision + recall), 1e-10)
        })
    
    # Mostrar resultados como tabla
    print("\n{:<15} {:<10} {:<15} {:<10} {:<10} {:<10}".format(
        "Método", "Umbral", "Anomalías", "Precisión", "Recall", "F1"))
    print("-" * 70)
    
    for r in results:
        print("{:<15} {:<10.4f} {:<15d} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            r['method'], r['threshold'], r['anomalies_found'], 
            r['precision'], r['recall'], r['f1']))
    
    # Demostración de umbral contextual con múltiples regímenes
    print("\nDemostración de umbral contextual con múltiples regímenes:")
    contextual_demo = ContextualThreshold("ContextualDemo")
    contextual_demo.add_regime("normal", 0.5)
    contextual_demo.add_regime("low_activity", 0.3)
    contextual_demo.add_regime("high_activity", 0.8)
    
    regimes = ["normal", "low_activity", "high_activity", "normal"]
    for regime in regimes:
        contextual_demo.set_regime(regime)
        print(f"Régimen: {regime}, Umbral: {contextual_demo.get_threshold():.4f}")
    
    # Ejemplo de feedback basado en validación externa
    print("\nEjemplo de umbral con retroalimentación:")
    meta = manager.thresholds["meta_threshold"]
    
    # Simular algunas validaciones (retroalimentación)
    meta.register_feedback(was_anomaly=True, predicted_anomaly=True)   # Verdadero positivo
    meta.register_feedback(was_anomaly=True, predicted_anomaly=False)  # Falso negativo
    meta.register_feedback(was_anomaly=False, predicted_anomaly=True)  # Falso positivo
    meta.register_feedback(was_anomaly=False, predicted_anomaly=True)  # Falso positivo
    
    # Forzar un ajuste basado en el feedback
    meta._adjust_weights()
    print("Después de retroalimentación, pesos ajustados:")
    for detector_name, weight in meta.threshold_weights.items():
        print(f"  {detector_name}: {weight:.4f}")
    
    print("\nEjemplo completado.")

