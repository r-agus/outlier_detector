#!/usr/bin/env python3
"""
Módulo de configuración centralizada para el sistema de detección de anomalías.

Este módulo define las configuraciones para cada componente del sistema, según lo descrito en el README:
- Ingesta y Preprocesamiento
- Modelado y Detección
- Umbrales Adaptativos
- Adaptación y Aprendizaje
- Procesamiento y Distribución
- Análisis y Visualización

"""

from dataclasses import dataclass, field, asdict
import os
from typing import Dict, List, Any
import json

# ---------------------------
# Configuración de Ingesta
# ---------------------------
@dataclass
class DataIngestionConfig:
    """
    Configuración para la capa de Ingesta de datos.
    """
    realtime_host: str = "localhost"           # Host para la recepción de datos en tiempo real
    realtime_port: int = 8000                  # Puerto asociado al colector de datos
    buffer_size: int = 1024                    # Tamaño del buffer para almacenamiento temporal
    detect_regime_changes: bool = True         # Habilitar la detección de cambios de régimen


# ---------------------------
# Configuración de Preprocesamiento
# ---------------------------
@dataclass
class PreprocessingConfig:
    """
    Configuración para la normalización y preparación de datos.
    """
    normalization_method: str = "minmax"       # Métodos: "minmax", "zscore", etc.
    window_size: int = 100                     # Tamaño de la ventana para análisis y estadísticas móviles


# ---------------------------
# Configuración de Modelado y Detección
# ---------------------------
@dataclass
class ModelConfig:
    """
    Configuración de los modelos de detección.
    """
    lof: Dict[str, Any] = field(default_factory=lambda: {"n_neighbors": 20, "contamination": 0.1})
    dbscan: Dict[str, Any] = field(default_factory=lambda: {"eps": 0.5, "min_samples": 5})
    autoencoder: Dict[str, Any] = field(default_factory=lambda: {
        "layers": [64, 32, 16, 32, 64],
        "activation": "relu",
        "epochs": 50,
        "batch_size": 32
    })
    isolation_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": 0.1
    })
    one_class_svm: Dict[str, Any] = field(default_factory=lambda: {
        "kernel": "rbf",
        "nu": 0.1,
        "gamma": "scale"
    })


# ---------------------------
# Configuración de Umbrales Adaptativos
# ---------------------------
@dataclass
class ThresholdConfig:
    """
    Configuración para la implementación de umbrales adaptativos.
    """
    moving_stats: Dict[str, Any] = field(default_factory=lambda: {"window_size": 100, "multiplier": 2.5})
    probabilistic: Dict[str, Any] = field(default_factory=lambda: {"percentile": 99})
    contextual: Dict[str, Any] = field(default_factory=lambda: {
        "default_threshold": 0.5,
        "regime_thresholds": {
            "normal": 0.5,
            "peak": 0.7,
            "off_peak": 0.4
        }
    })
    meta: Dict[str, Any] = field(default_factory=lambda: {"monitor_interval": 60, "adjustment_factor": 0.1})
    
    # Mapping de regímenes a estrategias de umbral
    regime_threshold_mapping: Dict[str, str] = field(default_factory=lambda: {
        "high_activity": "probabilistic",
        "low_activity": "moving_stats",
        "normal": "meta_threshold",
        # Otros regímenes pueden añadirse aquí con sus estrategias correspondientes
        "default": "meta_threshold"  # Estrategia por defecto para regímenes no mapeados
    })


# ---------------------------
# Configuración de Adaptación y Aprendizaje
# ---------------------------
@dataclass
class AdaptationConfig:
    """
    Configuración para mecanismos de adaptación automática y aprendizaje incremental.
    """
    sliding_window: Dict[str, Any] = field(default_factory=lambda: {"size": 100, "decay": 0.9})
    concept_drift: Dict[str, Any] = field(default_factory=lambda: {"ks_alpha": 0.05})
    incremental_learning: Dict[str, Any] = field(default_factory=lambda: {"mini_batch_size": 32, "learning_rate": 0.001})
    feedback: Dict[str, Any] = field(default_factory=lambda: {"enable": True, "feedback_weight": 0.5})


# ---------------------------
# Configuración de Procesamiento y Distribución
# ---------------------------
@dataclass
class ProcessingConfig:
    """
    Configuración para el equilibrio entre procesamiento en tiempo real y complejidad computacional.
    """
    model_stratification: Dict[str, List[str]] = field(default_factory=lambda: {
        "fast_layer": ["statistical"],
        "medium_layer": ["unsupervised"],
        "complex_layer": ["deep_learning"]
    })
    parallelization: Dict[str, Any] = field(default_factory=lambda: {
        "num_threads": 4,
        "distributed_system": "spark"  # Opciones: "spark", "kafka", "none", etc.
    })


# ---------------------------
# Configuración de Análisis y Visualización
# ---------------------------
@dataclass
class VisualizationConfig:
    """
    Configuración para la visualización y almacenamiento histórico de datos.
    """
    monitoring_panel: Dict[str, Any] = field(default_factory=lambda: {"refresh_interval": 1})  # en segundos
    historical_storage: Dict[str, Any] = field(default_factory=lambda: {
        "storage_path": os.path.join(os.getcwd(), "data", "historical"),
        "retention_days": 30
    })


# ---------------------------
# Configuración Global
# ---------------------------
@dataclass
class Config:
    """
    Clase principal de configuración que centraliza todas las configuraciones del sistema.
    """
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def update_from_dict(self, config_updates: Dict[str, Any]):
        """
        Actualiza la configuración a partir de un diccionario.
        Permite modificar de forma dinámica la configuración del sistema.
        """
        for section, updates in config_updates.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if isinstance(updates, dict):
                    for key, value in updates.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                        else:
                            # Si el atributo no existe, se puede extender la configuración
                            if isinstance(section_obj, dict):
                                section_obj[key] = value
                else:
                    setattr(self, section, updates)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración completa a un diccionario.
        """
        return asdict(self)


def load_config_from_json(file_path: str) -> Config:
    """
    Carga configuraciones desde un archivo JSON y actualiza la configuración global.

    :param file_path: Ruta al archivo JSON de configuración.
    :return: Instancia de Config actualizada.
    """
    with open(file_path, "r") as f:
        config_data = json.load(f)
    config_instance = Config()
    config_instance.update_from_dict(config_data)
    return config_instance


# Instancia global de configuración, accesible desde el resto del sistema
config = Config()

if __name__ == "__main__":
    # Impresión de la configuración actual para verificación
    import pprint
    pprint.pprint(config.to_dict())
