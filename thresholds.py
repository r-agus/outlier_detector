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
            "high_activity": 0.7,
            "low_activity": 0.4
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
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import pandas as pd
        from sklearn.metrics import precision_recall_curve, auc, roc_curve
        import seaborn as sns
        
        print("Demostración avanzada de umbrales adaptativos para detección de anomalías")
        print("=" * 80)
        
        # Generación de datos sintéticos complejos con múltiples regímenes
        print("\nGenerando datos sintéticos con múltiples regímenes y tipos de anomalías...")
        np.random.seed(42)
        
        # Parámetros para la generación de datos
        n_samples = 1000
        regime_changes = [200, 500, 800]  # Puntos donde cambia el régimen
        
        # Generar datos base (distribución normal)
        base_data = np.random.randn(n_samples)
        
        # Añadir tendencias y cambios de régimen
        data = base_data.copy()
        regimes = np.zeros(n_samples, dtype=int)
        
        # Régimen 1: Normal (0-200)
        data[:regime_changes[0]] = data[:regime_changes[0]] * 0.5
        
        # Régimen 2: Actividad alta (200-500)
        data[regime_changes[0]:regime_changes[1]] = data[regime_changes[0]:regime_changes[1]] * 1.5 + 1
        regimes[regime_changes[0]:regime_changes[1]] = 1
        
        # Régimen 3: Actividad baja con ruido (500-800)
        data[regime_changes[1]:regime_changes[2]] = data[regime_changes[1]:regime_changes[2]] * 0.3 - 0.5
        regimes[regime_changes[1]:regime_changes[2]] = 2
        
        # Régimen 4: Volver a normal pero con más varianza (800-1000)
        data[regime_changes[2]:] = data[regime_changes[2]:] * 0.7
        regimes[regime_changes[2]:] = 3
        
        # Añadir componente estacional
        seasonal_component = 0.5 * np.sin(np.linspace(0, 10 * np.pi, n_samples))
        data = data + seasonal_component
        
        # Insertar anomalías de diferentes tipos
        anomaly_indices = []
        
        # Tipo 1: Valores extremos aislados
        spike_indices = [50, 150, 350, 650, 950]
        for idx in spike_indices:
            data[idx] = data[idx] + 4.0 if np.random.rand() > 0.5 else data[idx] - 4.0
            anomaly_indices.append(idx)
            
        # Tipo 2: Secuencias anómalas (cambios de nivel)
        sequence_starts = [250, 550, 850]
        for start in sequence_starts:
            length = np.random.randint(5, 15)
            shift = 3.0 if np.random.rand() > 0.5 else -3.0
            data[start:start+length] = data[start:start+length] + shift
            anomaly_indices.extend(range(start, start+length))
            
        # Tipo 3: Cambios en varianza
        variance_starts = [100, 400, 700]
        for start in variance_starts:
            length = np.random.randint(10, 20)
            data[start:start+length] = data[start:start+length] * 3.0
            anomaly_indices.extend(range(start, start+length))
            
        # Convertir a array y eliminar duplicados
        anomaly_indices = np.unique(anomaly_indices)
        anomaly_labels = np.zeros(n_samples)
        anomaly_labels[anomaly_indices] = 1
        
        # Calcular puntuaciones de anomalía simuladas (ejemplo simple: distancia absoluta desde la media móvil)
        window = 50
        all_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            if i < window:
                window_start = 0
            else:
                window_start = i - window
                
            window_mean = np.mean(data[window_start:i+1])
            all_scores[i] = abs(data[i] - window_mean)
            
        # Normalizar puntuaciones para que estén entre 0 y 1
        all_scores = (all_scores - np.min(all_scores)) / (np.max(all_scores) - np.min(all_scores))
        
        # Crear gestor de umbrales
        print("Inicializando y configurando estrategias de umbral...")
        manager = ThresholdManager()
        
        # Crear diversas estrategias de umbral
        # static_threshold = StaticThreshold("static", threshold_value=0.6)
        moving_stats = MovingStatsThreshold("moving_stats")
        probabilistic = ProbabilisticThreshold("probabilistic")
        
        # Crear umbral contextual para regímenes
        contextual = ContextualThreshold("contextual")
        
        # Definir mapeo de regímenes a nombres consistentes con config.py
        # Los regímenes en nuestros datos están como números (0, 1, 2, 3)
        # Pero queremos usar nombres semánticos consistentes con la configuración
        regime_mapping = {
            0: "normal",             # Régimen 0: normal
            1: "high_activity",      # Régimen 1: alta actividad 
            2: "low_activity",       # Régimen 2: baja actividad
            3: "transition"          # Régimen 3: transición (nuevo régimen)
        }
        
        # Añadir el nuevo régimen al umbral contextual
        # Todos los demás regímenes ya deberían existir en la configuración predeterminada
        contextual.add_regime("transition", 0.55)  # Régimen de transición: umbral adaptado
        
        # Añadir umbrales al gestor
        # manager.add_threshold(static_threshold)
        manager.add_threshold(moving_stats)
        manager.add_threshold(probabilistic)
        manager.add_threshold(contextual)
        
        # Configurar meta-umbral
        manager.setup_meta_threshold(
            threshold_names=["moving_stats", "probabilistic", "contextual"],
            weights=[1.0, 1.0, 1.5]
        )
        
        # Simular procesamiento de datos en tiempo real y actualización de umbrales
        print("Simulando procesamiento de datos en tiempo real...")
        
        # Almacenar valores de umbral para cada método a lo largo del tiempo
        threshold_values = {
            name: np.zeros(n_samples) for name in manager.thresholds.keys()
        }
        
        # Procesar cada punto de datos secuencialmente
        for i in range(n_samples):
            # Convertir el valor numérico del régimen al nombre semántico correspondiente
            current_regime_id = regimes[i]
            current_regime_name = regime_mapping[current_regime_id]
            
            # Establecer régimen actual para el umbral contextual usando el nombre semántico
            contextual.set_regime(current_regime_name)
            
            # Actualizar todos los umbrales con el nuevo punto
            score_point = np.array([all_scores[i]])
            manager.update(score_point)
            
            # Almacenar valores de umbral actuales
            for name, threshold in manager.thresholds.items():
                threshold_values[name][i] = threshold.get_threshold()
        
        # Comparar rendimiento de los diferentes métodos
        print("\nEvaluando rendimiento de diferentes estrategias de umbral:")
        
        # Crear figura compleja para visualización
        plt.figure(figsize=(15, 20))
        gs = GridSpec(4, 2, height_ratios=[2, 1, 1, 1])
        
        # 1. Datos originales con anomalías marcadas
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(data, 'b-', alpha=0.7, label='Datos')
        ax1.scatter(anomaly_indices, data[anomaly_indices], color='red', s=50, marker='x', label='Anomalías reales')
        
        # Marcar cambios de régimen con líneas verticales
        for change in regime_changes:
            ax1.axvline(x=change, color='green', linestyle='--', alpha=0.7)
            
        ax1.set_title('Datos sintéticos con múltiples regímenes y anomalías', fontsize=14)
        ax1.set_xlabel('Muestra')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Anotar los regímenes
        regime_names = ['Normal', 'Alta actividad', 'Baja actividad', 'Normal con varianza']
        regime_start = [0] + regime_changes
        for i in range(len(regime_names)):
            mid_point = (regime_start[i] + (regime_start[i+1] if i < len(regime_start)-1 else n_samples)) // 2
            ax1.text(mid_point, ax1.get_ylim()[1] * 0.9, f'Régimen: {regime_names[i]}', 
                    horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # 2. Puntuaciones de anomalía y umbrales
        ax2 = plt.subplot(gs[1, :])
        ax2.plot(all_scores, 'b-', label='Puntuación de anomalía')
        
        # Añadir líneas para cada umbral
        for name, values in threshold_values.items():
            ax2.plot(values, label=f'Umbral: {name}', alpha=0.7, linestyle='--')
            
        ax2.set_title('Puntuaciones de anomalía y umbrales adaptativos', fontsize=14)
        ax2.set_xlabel('Muestra')
        ax2.set_ylabel('Puntuación/Umbral')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Precisión-Recall para cada método
        ax3 = plt.subplot(gs[2, 0])
        
        # Calculate a single precision-recall curve using the common anomaly scores
        precision, recall, pr_thresholds = precision_recall_curve(anomaly_labels, all_scores)
        pr_auc = auc(recall, precision)
        
        # Plot the precision-recall curve once - this is common for all methods
        ax3.plot(recall, precision, 'b-', label=f'PR Curve (AUC={pr_auc:.3f})')
        
        # Calculate results for each threshold method and mark their position on the curve
        results = []
        markers = ['o', 's', 'D', '^', 'v']  # Different marker styles
        colors = ['red', 'green', 'orange', 'purple', 'cyan']  # Different colors
        
        for i, (name, threshold_vals) in enumerate(threshold_values.items()):
            # For each threshold method, calculate its performance metrics
            predictions = all_scores > threshold_vals  # Apply method's thresholds
            
            # Calculate metrics using the thresholds from this method
            true_positives = np.sum(predictions & (anomaly_labels == 1))
            false_positives = np.sum(predictions & (anomaly_labels == 0))
            false_negatives = np.sum((~predictions) & (anomaly_labels == 1))
            
            precision_val = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall_val = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
            
            # Mark the threshold position on the PR curve
            marker_idx = i % len(markers)
            color_idx = i % len(colors)
            ax3.scatter(recall_val, precision_val, marker=markers[marker_idx], color=colors[color_idx], 
                       s=100, label=f'{name} (F1={f1:.3f})')
            
            # Store results for the table
            results.append({
                'method': name,
                'threshold': threshold_vals[-1],
                'anomalies_found': np.sum(predictions),
                'precision': precision_val,
                'recall': recall_val,
                'f1': f1,
                'pr_auc': pr_auc  # Same AUC for all since it's the same curve
            })
        
        ax3.set_title('Curva Precision-Recall y Posición de Umbrales', fontsize=14)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='lower left')
        
        # 4. ROC para cada método
        ax4 = plt.subplot(gs[2, 1])
        
        # Calculate a single ROC curve using the common anomaly scores
        fpr, tpr, roc_thresholds = roc_curve(anomaly_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve once
        ax4.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC={roc_auc:.3f})')
        
        # Mark each threshold method's position on the ROC curve
        for i, (name, threshold_vals) in enumerate(threshold_values.items()):
            # Calculate the last FPR and TPR for this method
            predictions = all_scores > threshold_vals[-1]
            
            # True positive rate and false positive rate
            tpr_val = np.sum(predictions & (anomaly_labels == 1)) / np.sum(anomaly_labels == 1)
            fpr_val = np.sum(predictions & (anomaly_labels == 0)) / np.sum(anomaly_labels == 0)
            
            # Mark the point on the ROC curve
            marker_idx = i % len(markers)
            color_idx = i % len(colors)
            ax4.scatter(fpr_val, tpr_val, marker=markers[marker_idx], color=colors[color_idx], 
                      s=100, label=f'{name}')
            
            # Update results with ROC metrics
            for result in results:
                if result['method'] == name:
                    result['tpr'] = tpr_val
                    result['fpr'] = fpr_val
                    break
                    
        ax4.plot([0, 1], [0, 1], 'k--', label='Random')
        ax4.set_title('Curva ROC y Posición de Umbrales', fontsize=14)
        ax4.set_xlabel('Tasa de Falsos Positivos')
        ax4.set_ylabel('Tasa de Verdaderos Positivos')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='lower right')
        
        # 5. Análisis de sensibilidad a cambios de régimen
        ax5 = plt.subplot(gs[3, 0])
        sensitivity_data = {}
        
        for name, values in threshold_values.items():
            regime_response = []
            
            # Medir tiempo de respuesta a cambios de régimen
            for change in regime_changes:
                # Medir cuánto tarda el umbral en estabilizarse tras cambio
                if change < 10:
                    continue
                
                baseline = values[change-10:change].mean()
                stabilized = False
                for i in range(30):  # Mirar 30 puntos después del cambio
                    if i + change >= len(values):
                        break
                    current = values[change+i]
                    # Verification to avoid division by zero or very small values
                    if baseline != 0 and abs(current - baseline) / max(abs(baseline), 0.001) < 0.1:
                        regime_response.append(i)
                        stabilized = True
                        break
                
                # If never stabilized, use max value (30)
                if not stabilized:
                    regime_response.append(30)
                        
            sensitivity_data[name] = regime_response
        
        # Visualizar tiempos de respuesta como diagrama de barras
        methods = []
        response_times = []
        
        for name, times in sensitivity_data.items():
            if times:
                methods.append(name)
                response_times.append(np.mean(times))
        
        # Check if we have data to plot
        if methods and response_times:
            ax5.bar(methods, response_times)
            ax5.set_title('Tiempo medio de respuesta a cambios de régimen', fontsize=14)
            ax5.set_xlabel('Método')
            ax5.set_ylabel('Muestras hasta estabilización')
        else:
            ax5.text(0.5, 0.5, "Datos insuficientes para análisis de respuesta", 
                    horizontalalignment='center', verticalalignment='center')
        
        # 6. Análisis de estabilidad de umbral
        ax6 = plt.subplot(gs[3, 1])
        
        # Calcular varianza móvil del umbral para evaluar estabilidad
        window_size = 20
        stability_data = {}
        
        for name, values in threshold_values.items():
            if len(values) <= window_size:
                continue
                
            variances = []
            for i in range(window_size, len(values)):
                window = values[i-window_size:i]
                variances.append(np.var(window))
            
            if variances:
                stability_data[name] = np.mean(variances)
            
        # Visualizar estabilidad como diagrama de barras
        methods = list(stability_data.keys())
        variance_values = list(stability_data.values())
        
        if methods and variance_values:
            ax6.bar(methods, variance_values)
            ax6.set_title('Estabilidad de umbrales (varianza promedio)', fontsize=14)
            ax6.set_xlabel('Método')
            ax6.set_ylabel('Varianza media')
        else:
            ax6.text(0.5, 0.5, "Datos insuficientes para análisis de estabilidad", 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig("adaptive_thresholds_analysis.png", dpi=300, bbox_inches='tight')
        print("\nAnálisis completo guardado como 'adaptive_thresholds_analysis.png'")
        
        # Mostrar resultados comparativos como tabla
        print("\nResultados comparativos:")
        results_df = pd.DataFrame(results)
        
        # Ordenar por F1-score descendente
        results_df = results_df.sort_values('f1', ascending=False)
        
        print("\n{:<15} {:<10} {:<15} {:<10} {:<10} {:<10} {:<10}".format(
            "Método", "Umbral", "Anomalías", "Precisión", "Recall", "F1", "AUC PR"))
        print("-" * 85)
        
        for _, r in results_df.iterrows():
            print("{:<15} {:<10.4f} {:<15d} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                r['method'], r['threshold'], r['anomalies_found'], 
                r['precision'], r['recall'], r['f1'], r['pr_auc']))
        
        # Análisis con diferentes métricas
        print("\nDemostración avanzada de umbral contextual con múltiples regímenes:")
        
        # Crear un nuevo umbral contextual para demostración
        contextual_demo = ContextualThreshold("ContextualDemoAdvanced")
        
        # Configurar los regímenes utilizando el método add_regime para cada régimen
        print("\n1. Configurando umbral contextual con mapeo de regímenes personalizado:")
        contextual_demo.add_regime("low_activity", 0.3)     # Régimen de baja actividad: umbral bajo
        contextual_demo.add_regime("high_activity", 0.7)    # Régimen de alta actividad: umbral alto
        contextual_demo.add_regime("normal", 0.45)          # Régimen normal: umbral moderado
        contextual_demo.add_regime("transition", 0.6)       # Régimen de transición: umbral intermedio
        
        print(f"   Mapeo configurado: {contextual_demo.regime_thresholds}")
        
        # Simular secuencia de cambios de régimen
        print("\n2. Simulando secuencia de cambios de régimen:")
        regime_sequence = [
            "normal", "normal", "normal", 
            "high_activity", "high_activity", 
            "low_activity", "low_activity", "low_activity", 
            "transition", "transition", 
            "normal", "normal"
        ]
        
        expected_thresholds = [contextual_demo.regime_thresholds[r] for r in regime_sequence]
        
        print("   Secuencia de regímenes:", regime_sequence)
        print("   Umbrales esperados:", [f"{t:.2f}" for t in expected_thresholds])
        
        # Verificar que los umbrales se ajusten correctamente
        actual_thresholds = []
        for regime in regime_sequence:
            contextual_demo.set_regime(regime)
            actual_thresholds.append(contextual_demo.get_threshold())
            
        print("   Umbrales obtenidos:", [f"{t:.2f}" for t in actual_thresholds])
        
        # Demostrar capacidad de respuesta a cambios rápidos
        print("\n3. Probando capacidad de respuesta a cambios rápidos de régimen:")
        moving_threshold = MovingStatsThreshold("moving_test")
        probabilistic_threshold = ProbabilisticThreshold("prob_test")
        
        # Simular un cambio rápido en los datos
        base_scores = np.random.normal(0, 1, 30)
        shift_scores = np.random.normal(3, 1, 20)  # Cambio significativo
        combined_scores = np.concatenate([base_scores, shift_scores])
        
        print("   Procesando 50 puntos con un cambio brusco en el punto 30...")
        
        # Procesar puntos secuencialmente
        contextual_values = []
        moving_values = []
        probabilistic_values = []
        
        for i, score in enumerate(combined_scores):
            # Actualizar umbrales
            point = np.array([score])
            
            # Usar nombres semánticos para los regímenes
            regime_name = "high_activity" if i >= 30 else "normal"
            contextual_demo.set_regime(regime_name)
            moving_threshold.update(point)
            probabilistic_threshold.update(point)
            
            # Guardar valores
            contextual_values.append(contextual_demo.get_threshold())
            moving_values.append(moving_threshold.get_threshold())
            probabilistic_values.append(probabilistic_threshold.get_threshold())
            
        print("   Tiempo de respuesta a cambio brusco (muestras):")
        print(f"   - Contextual: 0 (inmediato, basado en régimen)")
        
        # Calcular tiempo de respuesta para otros métodos
        moving_response = 0
        prob_response = 0
        
        for i in range(30, 50):
            if moving_response == 0 and moving_values[i] > moving_values[29] * 1.2:
                moving_response = i - 30 + 1
                
            if prob_response == 0 and probabilistic_values[i] > probabilistic_values[29] * 1.2:
                prob_response = i - 30 + 1
                
        print(f"   - Estadísticas móviles: {moving_response} muestras")
        print(f"   - Probabilístico: {prob_response} muestras")
        
    except Exception as e:
        print(f"No se pudo completar el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

