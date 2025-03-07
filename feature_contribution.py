#!/usr/bin/env python3
"""
Módulo para el análisis de contribución de características en anomalías detectadas.

Este módulo implementa técnicas para determinar qué variables/características
contribuyen más significativamente a la identificación de anomalías, proporcionando
explicabilidad a las detecciones realizadas por los modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import defaultdict
import logging
import time
from scipy import stats

# Importar modelos y configuración
from config import config
from models import BaseAnomalyDetector, AutoencoderDetector, IsolationForestDetector

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('feature_contribution')

class FeatureContributor:
    """
    Clase base para análisis de contribución de características.
    Proporciona métodos para identificar qué features contribuyen más a una anomalía.
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Inicializa el analizador de contribución de características.
        
        Args:
            feature_names: Nombres de las características para los reportes (opcional)
        """
        self.feature_names = feature_names
        logger.info(f"Inicializado analizador de contribución de características")
    
    def get_feature_name(self, index: int) -> str:
        """
        Obtiene el nombre de una característica según su índice.
        
        Args:
            index: Índice de la característica
            
        Returns:
            Nombre de la característica
        """
        if self.feature_names and index < len(self.feature_names):
            return self.feature_names[index]
        else:
            return f"Feature_{index}"
    
    def set_feature_names(self, names: List[str]) -> None:
        """
        Establece los nombres de las características.
        
        Args:
            names: Lista con los nombres de las características
        """
        self.feature_names = names
        logger.debug(f"Nombres de características establecidos: {names}")
    
    def calculate_z_scores(self, normal_data: np.ndarray, anomaly_data: np.ndarray) -> np.ndarray:
        """
        Calcula z-scores para cada característica comparando datos normales y anómalos.
        
        Args:
            normal_data: Matriz con datos normales
            anomaly_data: Matriz con datos anómalos
            
        Returns:
            Array con z-scores para cada característica
        """
        # Calcular estadísticas sobre datos normales
        normal_mean = np.mean(normal_data, axis=0)
        normal_std = np.std(normal_data, axis=0) + 1e-10  # Evitar división por cero
        
        # Para cada ejemplo anómalo, calcular cuánto se desvía de la normalidad
        z_scores = np.abs((anomaly_data - normal_mean) / normal_std)
        
        # Si hay múltiples ejemplos anómalos, promediar sus z-scores
        if z_scores.ndim > 1 and z_scores.shape[0] > 1:
            z_scores = np.mean(z_scores, axis=0)
            
        return z_scores
    
    def rank_features_by_contribution(self, contributions: np.ndarray) -> List[Tuple[str, float]]:
        """
        Ordena las características según su contribución a la anomalía.
        
        Args:
            contributions: Array con puntuaciones de contribución por característica
            
        Returns:
            Lista ordenada de tuplas (nombre_característica, puntuación)
        """
        # Crear lista de tuplas (índice, puntuación)
        feature_scores = [(i, score) for i, score in enumerate(contributions)]
        
        # Ordenar por puntuación descendente
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convertir a formato (nombre, puntuación)
        ranked_features = [(self.get_feature_name(i), score) for i, score in feature_scores]
        
        return ranked_features


class PerturbationAnalyzer(FeatureContributor):
    """
    Análisis de contribución mediante perturbación de características.
    Modifica cada característica individualmente y observa el impacto en la puntuación de anomalía.
    """
    
    def __init__(self, detector: BaseAnomalyDetector, feature_names: List[str] = None):
        """
        Inicializa el analizador de perturbación.
        
        Args:
            detector: Detector de anomalías entrenado
            feature_names: Nombres de las características
        """
        super().__init__(feature_names)
        self.detector = detector
        
        if not detector.is_fitted:
            raise ValueError("El detector debe estar entrenado antes de analizar contribuciones")
            
        logger.info(f"Inicializado analizador de perturbación con detector: {detector.name}")
    
    def analyze(self, anomaly_data: np.ndarray, normal_data: np.ndarray, 
               num_perturbations: int = 10) -> Dict[str, Any]:
        """
        Analiza la contribución de cada característica mediante perturbaciones.
        
        Args:
            anomaly_data: Datos anómalos a analizar
            normal_data: Datos normales para referencia
            num_perturbations: Número de perturbaciones a aplicar
            
        Returns:
            Diccionario con resultados del análisis
        """
        start_time = time.time()
        
        # Asegurar que los datos tengan forma adecuada
        if anomaly_data.ndim == 1:
            anomaly_data = anomaly_data.reshape(1, -1)
            
        # Calcular puntuación de anomalía inicial
        original_score = self.detector.decision_function(anomaly_data)[0]
        
        # Inicializar array para almacenar impacto de perturbaciones
        num_features = anomaly_data.shape[1]
        feature_impact = np.zeros(num_features)
        
        # Para cada característica, realizar perturbaciones
        for feature_idx in range(num_features):
            # Obtener valores normales para esta característica
            normal_values = normal_data[:, feature_idx]
            
            # Generar perturbaciones (valores entre min y max de datos normales)
            min_val, max_val = np.min(normal_values), np.max(normal_values)
            perturbation_values = np.linspace(min_val, max_val, num_perturbations)
            
            impact_scores = []
            
            # Probar cada perturbación
            for pert_value in perturbation_values:
                # Crear copia del dato anómalo
                perturbed_sample = anomaly_data.copy()
                # Aplicar perturbación a la característica específica
                perturbed_sample[0, feature_idx] = pert_value
                # Evaluar nuevo score
                new_score = self.detector.decision_function(perturbed_sample)[0]
                # Calcular impacto (reducción en score = mayor importancia)
                score_change = original_score - new_score
                impact_scores.append(score_change)
            
            # El impacto de la característica es el máximo cambio logrado
            feature_impact[feature_idx] = max(0, np.max(impact_scores))
        
        # Normalizar impactos para que sumen 1
        if np.sum(feature_impact) > 0:
            feature_impact = feature_impact / np.sum(feature_impact)
            
        # Ordenar características por impacto
        ranked_features = self.rank_features_by_contribution(feature_impact)
        
        # Preparar resultados
        results = {
            "feature_impact": feature_impact,
            "ranked_features": ranked_features,
            "original_score": original_score,
            "analysis_method": "perturbation",
            "analysis_time": time.time() - start_time
        }
        
        logger.info(f"Análisis de perturbación completado en {results['analysis_time']:.2f}s")
        
        return results
    
    def plot_feature_impact(self, results: Dict[str, Any], top_n: int = 10, 
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Visualiza el impacto de las características en la anomalía.
        
        Args:
            results: Resultados del análisis
            top_n: Número de características principales a mostrar
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        ranked_features = results["ranked_features"][:top_n]
        feature_names = [name for name, _ in ranked_features]
        impact_scores = [score for _, score in ranked_features]
        
        # Invertir orden para visualización
        feature_names.reverse()
        impact_scores.reverse()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras horizontales
        bars = ax.barh(feature_names, impact_scores, color='skyblue')
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f"{width:.3f}", ha='left', va='center')
        
        ax.set_title(f"Top {top_n} Características por Contribución a Anomalía")
        ax.set_xlabel("Impacto Relativo")
        ax.set_xlim(0, max(impact_scores) * 1.15)  # Espacio para etiquetas
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig


class ReconstructionErrorAnalyzer(FeatureContributor):
    """
    Análisis de contribución basado en errores de reconstrucción para autoencoders.
    Examina qué características tienen mayor error de reconstrucción.
    """
    
    def __init__(self, autoencoder_detector: AutoencoderDetector, feature_names: List[str] = None):
        """
        Inicializa el analizador de errores de reconstrucción.
        
        Args:
            autoencoder_detector: Detector basado en autoencoder
            feature_names: Nombres de las características
        """
        super().__init__(feature_names)
        
        # Verificar que el detector sea un autoencoder
        if not isinstance(autoencoder_detector, AutoencoderDetector):
            raise TypeError("Este analizador requiere un detector basado en autoencoder")
            
        self.detector = autoencoder_detector
        
        if not self.detector.is_fitted:
            raise ValueError("El autoencoder debe estar entrenado antes de analizar contribuciones")
            
        logger.info(f"Inicializado analizador de reconstrucción con detector: {autoencoder_detector.name}")
    
    def analyze(self, anomaly_data: np.ndarray, normal_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analiza la contribución de características basado en error de reconstrucción.
        
        Args:
            anomaly_data: Datos anómalos a analizar
            normal_data: Datos normales para comparación (opcional)
            
        Returns:
            Diccionario con resultados del análisis
        """
        start_time = time.time()
        
        # Asegurar que los datos tengan forma adecuada
        if anomaly_data.ndim == 1:
            anomaly_data = anomaly_data.reshape(1, -1)
        
        # 1. Reconstruir datos anómalos con el autoencoder
        reconstructed = self.detector.model.predict(anomaly_data)
        
        # 2. Calcular error de reconstrucción por característica
        reconstruction_errors = np.abs(anomaly_data - reconstructed)
        
        # Si hay múltiples ejemplos, promediar
        if reconstruction_errors.shape[0] > 1:
            feature_errors = np.mean(reconstruction_errors, axis=0)
        else:
            feature_errors = reconstruction_errors[0]
        
        # 3. Calcular error de reconstrucción normal (línea base) si se proporcionan datos normales
        if normal_data is not None:
            # Reconstrucción de datos normales
            normal_reconstructed = self.detector.model.predict(normal_data)
            normal_errors = np.abs(normal_data - normal_reconstructed)
            baseline_errors = np.mean(normal_errors, axis=0)
            
            # Comparar con línea base (cuánto mayor es el error en la anomalía)
            relative_errors = feature_errors / (baseline_errors + 1e-10)
            
            # Usar errores relativos como contribución
            feature_contribution = relative_errors
        else:
            # Usar errores absolutos como contribución
            feature_contribution = feature_errors
        
        # Normalizar puntuaciones para que sumen 1
        feature_contribution = feature_contribution / np.sum(feature_contribution)
        
        # Ordenar características por contribución
        ranked_features = self.rank_features_by_contribution(feature_contribution)
        
        # Preparar resultados
        results = {
            "feature_contribution": feature_contribution,
            "reconstruction_errors": feature_errors,
            "ranked_features": ranked_features,
            "analysis_method": "reconstruction_error",
            "analysis_time": time.time() - start_time
        }
        
        # Si se proporcionaron datos normales, incluir comparativa
        if normal_data is not None:
            results["baseline_errors"] = baseline_errors
            results["relative_errors"] = relative_errors
            
        logger.info(f"Análisis de error de reconstrucción completado en {results['analysis_time']:.2f}s")
        
        return results
    
    def plot_reconstruction_comparison(self, anomaly_data: np.ndarray, 
                                     results: Dict[str, Any],
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualiza la comparación entre datos originales y reconstruidos.
        
        Args:
            anomaly_data: Datos anómalos originales
            results: Resultados del análisis
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        if anomaly_data.ndim == 1:
            anomaly_data = anomaly_data.reshape(1, -1)
            
        # Reconstruir datos
        reconstructed = self.detector.model.predict(anomaly_data)
        
        # Preparar datos para visualización
        if anomaly_data.shape[0] > 1:
            # Si hay múltiples ejemplos, usar el primero para la visualización
            original = anomaly_data[0]
            recon = reconstructed[0]
        else:
            original = anomaly_data[0]
            recon = reconstructed[0]
        
        # Crear gráfico
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        
        # 1. Gráfico de valores originales vs. reconstruidos
        feature_indices = range(len(original))
        feature_labels = [self.get_feature_name(i) for i in feature_indices]
        
        axs[0].plot(feature_indices, original, 'b-', marker='o', label='Original')
        axs[0].plot(feature_indices, recon, 'r--', marker='x', label='Reconstruido')
        axs[0].set_title('Valores Originales vs. Reconstruidos')
        axs[0].set_xlabel('Característica')
        axs[0].set_ylabel('Valor')
        axs[0].set_xticks(feature_indices)
        axs[0].set_xticklabels(feature_labels, rotation=45, ha='right')
        axs[0].legend()
        axs[0].grid(True)
        
        # 2. Gráfico de errores de reconstrucción
        errors = np.abs(original - recon)
        bars = axs[1].bar(feature_indices, errors, color='skyblue')
        axs[1].set_title('Error de Reconstrucción por Característica')
        axs[1].set_xlabel('Característica')
        axs[1].set_ylabel('Error Absoluto')
        axs[1].set_xticks(feature_indices)
        axs[1].set_xticklabels(feature_labels, rotation=45, ha='right')
        
        # Resaltar las características con mayor error
        max_indices = np.argsort(errors)[-3:]  # Top 3 errores
        for idx in max_indices:
            bars[idx].set_color('red')
            axs[1].text(idx, errors[idx] + 0.01, f"{errors[idx]:.3f}", 
                       ha='center', va='bottom', fontweight='bold')
        
        axs[1].grid(True, axis='y')
        
        plt.tight_layout()
        return fig


class StatisticalContributionAnalyzer(FeatureContributor):
    """
    Análisis de contribución basado en estadísticas comparativas.
    Compara distribuciones estadísticas entre datos normales y anómalos.
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Inicializa el analizador estadístico.
        
        Args:
            feature_names: Nombres de las características
        """
        super().__init__(feature_names)
        logger.info("Inicializado analizador de contribución estadística")
    
    def analyze(self, anomaly_data: np.ndarray, normal_data: np.ndarray, 
               method: str = 'z_score') -> Dict[str, Any]:
        """
        Analiza la contribución de características mediante comparación estadística.
        
        Args:
            anomaly_data: Datos anómalos a analizar
            normal_data: Datos normales para referencia
            method: Método de análisis ('z_score', 'ks_test', 'mahalanobis')
            
        Returns:
            Diccionario con resultados del análisis
        """
        start_time = time.time()
        
        # Asegurar formato adecuado
        if anomaly_data.ndim == 1:
            anomaly_data = anomaly_data.reshape(1, -1)
        
        # Diferentes métodos de análisis
        if method == 'z_score':
            # Z-score: qué tan lejos está cada característica de su distribución normal
            contributions = self.calculate_z_scores(normal_data, anomaly_data)
            
        elif method == 'ks_test':
            # Test Kolmogorov-Smirnov: diferencia entre distribuciones
            contributions = np.zeros(anomaly_data.shape[1])
            
            for i in range(anomaly_data.shape[1]):
                # Para que tenga sentido estadístico, necesitamos suficientes muestras anómalas
                if anomaly_data.shape[0] >= 5:
                    statistic, _ = stats.ks_2samp(anomaly_data[:, i], normal_data[:, i])
                    contributions[i] = statistic
                else:
                    # Con pocas muestras, comparar con percentiles
                    anomaly_val = np.mean(anomaly_data[:, i])
                    percentile = stats.percentileofscore(normal_data[:, i], anomaly_val)
                    # Convertir a un valor entre 0 y 1, donde 1 es más anómalo
                    p_score = max(percentile, 100 - percentile) / 100
                    contributions[i] = p_score
        
        elif method == 'mahalanobis':
            # Distancia de Mahalanobis: distancia multivariada considerando correlaciones
            normal_mean = np.mean(normal_data, axis=0)
            cov_matrix = np.cov(normal_data, rowvar=False)
            
            # Invertir matriz de covarianza (con regularización si es necesario)
            try:
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Si la matriz no es invertible, añadir pequeña regularización
                cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
                inv_cov = np.linalg.inv(cov_matrix)
            
            # Calcular contribución de cada característica a la distancia de Mahalanobis
            contributions = np.zeros(anomaly_data.shape[1])
            
            for i in range(anomaly_data.shape[1]):
                # Crear vector de contribuciones dejando solo esta característica
                contrib_vector = np.zeros_like(normal_mean)
                contrib_vector[i] = anomaly_data[0, i] - normal_mean[i]
                
                # Calcular contribución parcial a la distancia de Mahalanobis
                partial_d = contrib_vector.T @ inv_cov @ contrib_vector
                contributions[i] = partial_d
        
        else:
            raise ValueError(f"Método de análisis '{method}' no reconocido")
        
        # Normalizar puntuaciones para que sumen 1
        if np.sum(contributions) > 0:
            contributions = contributions / np.sum(contributions)
        
        # Ordenar características por contribución
        ranked_features = self.rank_features_by_contribution(contributions)
        
        # Preparar resultados
        results = {
            "feature_contribution": contributions,
            "ranked_features": ranked_features,
            "analysis_method": method,
            "analysis_time": time.time() - start_time
        }
        
        logger.info(f"Análisis estadístico ({method}) completado en {results['analysis_time']:.2f}s")
        
        return results
    
    def plot_distribution_comparison(self, anomaly_data: np.ndarray, normal_data: np.ndarray,
                                   top_features: List[str], 
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compara distribuciones de características normales vs. anómalas.
        
        Args:
            anomaly_data: Datos anómalos
            normal_data: Datos normales
            top_features: Lista de nombres de características a visualizar
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        # Convertir nombres de características a índices
        if not self.feature_names:
            feature_indices = [int(name.split('_')[1]) for name in top_features]
        else:
            feature_indices = [self.feature_names.index(name) for name in top_features]
        
        n_features = len(top_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_features == 1:
            axs = np.array([axs])
        
        axs = axs.flatten()
        
        for i, (idx, ax) in enumerate(zip(feature_indices, axs)):
            # Extraer valores para esta característica
            normal_vals = normal_data[:, idx]
            anomaly_vals = anomaly_data[:, idx]
            
            # Crear histograma/KDE para datos normales
            sns.histplot(normal_vals, ax=ax, color='blue', label='Normal', 
                        alpha=0.6, kde=True, stat='density')
            
            # Superponer valores anómalos como rug plot o histograma según cantidad
            if len(anomaly_vals) < 10:
                # Con pocos valores, mostrar marcas en eje x
                for val in anomaly_vals:
                    ax.axvline(x=val, color='red', linestyle='--', alpha=0.7)
                # Texto adicional para entender los valores
                anomaly_mean = np.mean(anomaly_vals)
                ax.text(0.05, 0.95, f'Valor anómalo: {anomaly_mean:.3f}', 
                       transform=ax.transAxes, color='red',
                       verticalalignment='top')
            else:
                # Con muchos valores, mostrar histograma/KDE
                sns.histplot(anomaly_vals, ax=ax, color='red', label='Anómalo', 
                            alpha=0.6, kde=True, stat='density')
            
            # Configurar eje
            ax.set_title(top_features[i])
            ax.set_xlabel('')
            ax.legend()
        
        # Ocultar ejes sin usar
        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)
        
        plt.suptitle('Comparación de Distribuciones: Normal vs. Anómalo', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Espacio para título principal
        
        return fig


class ModelSpecificAnalyzer(FeatureContributor):
    """
    Analizador de contribución específico para ciertos tipos de modelos.
    Aprovecha las capacidades intrínsecas de los modelos para determinar importancia de características.
    """
    
    def __init__(self, detector: BaseAnomalyDetector, feature_names: List[str] = None):
        """
        Inicializa el analizador específico de modelo.
        
        Args:
            detector: Detector de anomalías
            feature_names: Nombres de las características
        """
        super().__init__(feature_names)
        self.detector = detector
        
        if not detector.is_fitted:
            raise ValueError("El detector debe estar entrenado antes de analizar contribuciones")
            
        logger.info(f"Inicializado analizador específico para modelo: {detector.name} ({type(detector).__name__})")
    
    def analyze(self, anomaly_data: np.ndarray, normal_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analiza contribuciones de características según el tipo específico de modelo.
        
        Args:
            anomaly_data: Datos anómalos a analizar
            normal_data: Datos normales para comparación (opcional)
            
        Returns:
            Diccionario con resultados del análisis
        """
        start_time = time.time()
        
        # Diferentes estrategias según el tipo de modelo
        if isinstance(self.detector, IsolationForestDetector):
            # Para Isolation Forest, usar importancia de características
            contributions = self._analyze_isolation_forest()
            method = "isolation_forest_importance"
            
        elif isinstance(self.detector, AutoencoderDetector):
            # Para Autoencoder, usar analizador de reconstrucción
            recon_analyzer = ReconstructionErrorAnalyzer(self.detector, self.feature_names)
            return recon_analyzer.analyze(anomaly_data, normal_data)
            
        else:
            # Para otros modelos, usar análisis genérico de perturbación
            if normal_data is not None:
                perturb_analyzer = PerturbationAnalyzer(self.detector, self.feature_names)
                return perturb_analyzer.analyze(anomaly_data, normal_data)
            else:
                # Sin datos normales, no podemos usar perturbación
                # Método alternativo: usar valores absolutos del punto anómalo como proxy
                if anomaly_data.ndim == 1:
                    contributions = np.abs(anomaly_data)
                else:
                    contributions = np.mean(np.abs(anomaly_data), axis=0)
                    
                method = "absolute_value_proxy"
        
        # Normalizar puntuaciones
        if np.sum(contributions) > 0:
            contributions = contributions / np.sum(contributions)
            
        # Ordenar características por contribución
        ranked_features = self.rank_features_by_contribution(contributions)
        
        # Preparar resultados
        results = {
            "feature_contribution": contributions,
            "ranked_features": ranked_features,
            "analysis_method": method,
            "analysis_time": time.time() - start_time,
            "detector_type": type(self.detector).__name__
        }
        
        logger.info(f"Análisis específico para {type(self.detector).__name__} completado en {results['analysis_time']:.2f}s")
        
        return results
    
    def _analyze_isolation_forest(self) -> np.ndarray:
        """
        Extrae importancia de características de un modelo Isolation Forest.
        
        Returns:
            Array con importancia relativa de cada característica
        """
        # Acceder al modelo interno de scikit-learn
        iforest_model = self.detector.model
        
        # Obtener número de nodos que utilizan cada característica
        n_nodes = np.zeros(iforest_model.n_features_in_, dtype=np.float64)
        
        # Recorrer cada árbol en el bosque
        for tree in iforest_model.estimators_:
            # Extraer estructura del árbol
            tree_struct = tree.tree_
            # Nodos no terminales (tienen característica asociada)
            non_terminal = tree_struct.children_left != -1
            # Incrementar contador para cada característica utilizada
            n_nodes += np.bincount(
                tree_struct.feature[non_terminal],
                minlength=iforest_model.n_features_in_
            )
        
        # Normalizar para obtener importancia relativa
        if np.sum(n_nodes) > 0:
            feature_importance = n_nodes / np.sum(n_nodes)
        else:
            feature_importance = np.ones(iforest_model.n_features_in_) / iforest_model.n_features_in_
            
        return feature_importance

    def plot_model_specific_analysis(self, results: Dict[str, Any], 
                                    top_n: int = 10,
                                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualiza los resultados del análisis específico de modelo.
        
        Args:
            results: Resultados del análisis
            top_n: Número de características principales a mostrar
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        ranked_features = results["ranked_features"][:top_n]
        feature_names = [name for name, _ in ranked_features]
        contributions = [score for _, score in ranked_features]
        
        # Invertir orden para mejor visualización
        feature_names.reverse()
        contributions.reverse()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras horizontales
        bars = ax.barh(feature_names, contributions, color='lightgreen')
        
        # Resaltar la característica más importante
        bars[0].set_color('darkgreen')
        
        # Añadir valores
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f"{width:.3f}", va='center')
        
        # Personalizar según el tipo de modelo
        if isinstance(self.detector, IsolationForestDetector):
            title = f"Importancia de características en Isolation Forest"
        else:
            title = f"Contribución de características para {type(self.detector).__name__}"
            
        ax.set_title(title)
        ax.set_xlabel("Importancia relativa")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig


class EnsembleFeatureAnalyzer(FeatureContributor):
    """
    Combina múltiples técnicas de análisis de características.
    Más robusto que utilizar una única técnica.
    """
    
    def __init__(self, detector: BaseAnomalyDetector, feature_names: List[str] = None):
        """
        Inicializa el analizador de ensamble.
        
        Args:
            detector: Detector de anomalías
            feature_names: Nombres de las características
        """
        super().__init__(feature_names)
        self.detector = detector
        
        # Crear diferentes analizadores según el tipo de detector
        self.analyzers = []
        
        # Siempre incluir análisis estadístico
        self.statistical_analyzer = StatisticalContributionAnalyzer(feature_names)
        self.analyzers.append(self.statistical_analyzer)
        
        # Analizadores específicos por tipo de modelo
        if isinstance(detector, AutoencoderDetector):
            self.recon_analyzer = ReconstructionErrorAnalyzer(detector, feature_names)
            self.analyzers.append(self.recon_analyzer)
        else:
            # Para otros detectores, usar analizador específico de modelo
            self.model_analyzer = ModelSpecificAnalyzer(detector, feature_names)
            self.analyzers.append(self.model_analyzer)
        
        # Analizador de perturbación para todos los modelos
        self.perturb_analyzer = PerturbationAnalyzer(detector, feature_names)
        self.analyzers.append(self.perturb_analyzer)
        
        logger.info(f"Analizador de ensamble inicializado con {len(self.analyzers)} técnicas de análisis")
    
    def analyze(self, anomaly_data: np.ndarray, normal_data: np.ndarray) -> Dict[str, Any]:
        """
        Realiza análisis combinado de múltiples técnicas.
        
        Args:
            anomaly_data: Datos anómalos a analizar
            normal_data: Datos normales para referencia
            
        Returns:
            Diccionario con resultados del análisis
        """
        start_time = time.time()
        
        # Resultados de cada analizador individual
        individual_results = []
        
        # Ejecutar cada analizador
        for analyzer in self.analyzers:
            if isinstance(analyzer, StatisticalContributionAnalyzer):
                # Para análisis estadístico
                result = analyzer.analyze(anomaly_data, normal_data)
            elif isinstance(analyzer, ReconstructionErrorAnalyzer):
                # Para análisis de reconstrucción
                result = analyzer.analyze(anomaly_data, normal_data)
            elif isinstance(analyzer, PerturbationAnalyzer):
                # Para análisis de perturbación
                result = analyzer.analyze(anomaly_data, normal_data)
            else:
                # Para análisis específico de modelo
                result = analyzer.analyze(anomaly_data, normal_data)
            
            individual_results.append(result)
        
        # Combinar puntuaciones de cada técnica
        combined_contribution = np.zeros(anomaly_data.shape[1])
        
        for result in individual_results:
            if "feature_contribution" in result:
                # Ensure consistent array shape by flattening/squeezing if needed
                feature_contrib = result["feature_contribution"]
                if feature_contrib.ndim > 1:
                    feature_contrib = feature_contrib.squeeze()
                combined_contribution += feature_contrib
        
        # Normalizar puntuación combinada
        if np.sum(combined_contribution) > 0:
            combined_contribution = combined_contribution / len(individual_results)
        
        # Ordenar características por contribución combinada
        ranked_features = self.rank_features_by_contribution(combined_contribution)
        
        # Preparar resultados
        results = {
            "feature_contribution": combined_contribution,
            "ranked_features": ranked_features,
            "individual_results": individual_results,
            "analysis_method": "ensemble",
            "analysis_time": time.time() - start_time
        }
        
        logger.info(f"Análisis de ensamble completado en {results['analysis_time']:.2f}s")
        
        return results
    
    def plot_ensemble_comparison(self, results: Dict[str, Any], 
                               top_n: int = 5,
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Visualiza una comparación de las diferentes técnicas de análisis.
        
        Args:
            results: Resultados del análisis de ensamble
            top_n: Número de principales características a mostrar
            figsize: Tamaño de la figura
            
        Returns:
            Objeto Figure de matplotlib
        """
        # Top características según el análisis combinado
        top_features = [name for name, _ in results["ranked_features"][:top_n]]
        
        # Extraer índices de estas características
        if self.feature_names:
            feature_indices = [self.feature_names.index(name) for name in top_features]
        else:
            feature_indices = [int(name.split('_')[1]) for name in top_features]
        
        # Extraer puntuaciones de cada método para las top características
        methods = []
        scores_by_method = {}
        
        # Check if individual_results exists in the results dictionary
        if "individual_results" not in results or not results["individual_results"]:
            logger.warning("No individual results found in ensemble results")
            # Create dummy data for visualization
            methods = ["dummy"]
            scores_by_method = {"dummy": [0.0] * len(top_features)}
        else:
            # Process individual results
            for individual in results["individual_results"]:
                method = individual.get("analysis_method", "unknown")
                methods.append(method)
                
                # Check if feature_contribution exists
                if "feature_contribution" not in individual:
                    logger.warning(f"No feature_contribution found for method {method}")
                    scores_by_method[method] = [0.0] * len(feature_indices)
                    continue
                
                # Extract scores
                try:
                    contrib = individual["feature_contribution"]
                    
                    # Handle different shapes and sizes
                    if contrib.ndim > 1:
                        # For multi-dimensional arrays, flatten or use first element
                        if contrib.shape[0] == 1:
                            contrib = contrib[0]
                        else:
                            contrib = np.mean(contrib, axis=0)
                    
                    # Check if array is large enough
                    max_index = max(feature_indices)
                    if len(contrib) <= max_index:
                        logger.warning(f"Contribution array for {method} is too small (size={len(contrib)})")
                        extended_contrib = np.zeros(max_index + 1)
                        extended_contrib[:len(contrib)] = contrib
                        contrib = extended_contrib
                    
                    # Get scores for top features
                    scores_by_method[method] = [contrib[idx] for idx in feature_indices]
                except Exception as e:
                    logger.error(f"Error extracting scores for {method}: {e}")
                    scores_by_method[method] = [0.0] * len(feature_indices)
        
        # Add combined scores
        methods.append("combined")
        
        # Check if feature_contribution exists in results
        if "feature_contribution" not in results:
            logger.warning("No combined feature_contribution found in results")
            combined = np.zeros(max(feature_indices) + 1)
        else:
            combined = results["feature_contribution"]
            
        # Safety check for combined scores
        try:
            max_index = max(feature_indices)
            if len(combined) <= max_index:
                extended_combined = np.zeros(max_index + 1)
                extended_combined[:len(combined)] = combined
                combined = extended_combined
                
            scores_by_method["combined"] = [combined[idx] for idx in feature_indices]
        except Exception as e:
            logger.error(f"Error extracting combined scores: {e}")
            scores_by_method["combined"] = [0.0] * len(feature_indices)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(top_features))
        width = 0.8 / len(methods) if methods else 0.4  # Handle empty methods list
        
        for i, method in enumerate(methods):
            offset = width * i - width * (len(methods) - 1) / 2
            bars = ax.bar(x + offset, scores_by_method[method], width, 
                         label=method.replace('_', ' ').title())
            
            # Highlight combined method
            if method == "combined":
                for bar in bars:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
        
        # Customize plot
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.set_title(f"Comparación de Métodos de Análisis para Top {top_n} Características")
        ax.set_ylabel("Importancia Relativa")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig


# Funciones de utilidad para análisis de contribución de características

def compare_feature_distributions(normal_data: np.ndarray, anomaly_data: np.ndarray, 
                               feature_names: List[str] = None,
                               top_n: int = 6,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Compara distribuciones de características entre datos normales y anómalos.
    
    Args:
        normal_data: Datos normales de referencia
        anomaly_data: Datos anómalos a analizar
        feature_names: Nombres de las características
        top_n: Número de características a visualizar
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    # Seleccionar las características más discriminativas
    analyzer = StatisticalContributionAnalyzer(feature_names)
    results = analyzer.analyze(anomaly_data, normal_data, method='ks_test')
    
    # Obtener top características
    top_features = results["ranked_features"][:top_n]
    
    # Crear figura para comparación
    fig, axes = plt.subplots(nrows=(top_n+1)//2, ncols=2, figsize=figsize)
    axes = axes.flatten()
    
    for i, (feature, score) in enumerate(top_features):
        if i >= len(axes):
            break
            
        # Obtener índice de esta característica
        if feature_names:
            idx = feature_names.index(feature)
        else:
            idx = int(feature.split('_')[1])
            
        # Extraer valores para esta característica
        normal_vals = normal_data[:, idx]
        
        # Para anomalía, puede ser un solo punto o varios
        if anomaly_data.ndim == 1:
            anomaly_vals = np.array([anomaly_data[idx]])
        elif anomaly_data.shape[0] == 1:
            anomaly_vals = np.array([anomaly_data[0, idx]])
        else:
            anomaly_vals = anomaly_data[:, idx]
        
        # Crear histograma para datos normales
        sns.histplot(normal_vals, ax=axes[i], color='blue', label='Normal', 
                    kde=True, stat="density", alpha=0.6)
        
        # Superponer valores anómalos
        if len(anomaly_vals) < 3:
            # Pocas anomalías: mostrar líneas verticales
            for val in anomaly_vals:
                axes[i].axvline(x=val, color='red', linestyle='--', linewidth=2)
                axes[i].text(val, 0, f"{val:.3f}", 
                            color='red', ha='center', va='bottom', fontsize=10)
        else:
            sns.histplot(anomaly_vals, ax=axes[i], color='red', 
                        label='Anomalía', kde=True, stat="density", alpha=0.5)
        
        # Configurar gráfico
        axes[i].set_title(f"{feature} (score={score:.3f})")
        axes[i].legend(loc="upper right")
        
    # Ocultar ejes no utilizados
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Comparación de Distribuciones: Normal vs. Anomalía", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig


def visualize_feature_correlation_impact(anomaly_data: np.ndarray, normal_data: np.ndarray, 
                                      feature_names: List[str] = None,
                                      top_n: int = 10,
                                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Visualiza cómo las correlaciones entre características contribuyen a anomalías.
    
    Args:
        anomaly_data: Datos anómalos
        normal_data: Datos normales para referencia
        feature_names: Nombres de las características
        top_n: Número de principales características a considerar
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib con matriz de correlación
    """
    # Ensure we have enough features for correlation analysis (at least 2)
    min_features = 2
    top_n = max(min_features, min(top_n, normal_data.shape[1]))
    
    # Identificar las características más importantes
    analyzer = StatisticalContributionAnalyzer(feature_names)
    results = analyzer.analyze(anomaly_data, normal_data)
    
    # Seleccionar top características
    top_indices = []
    for feature, _ in results["ranked_features"][:top_n]:
        if feature_names:
            try:
                idx = feature_names.index(feature)
                top_indices.append(idx)
            except ValueError:
                logger.warning(f"Feature name {feature} not found in feature_names")
        else:
            try:
                idx = int(feature.split('_')[1])
                top_indices.append(idx)
            except (IndexError, ValueError):
                logger.warning(f"Could not parse feature index from {feature}")
    
    # Ensure we have at least 2 features
    if len(top_indices) < min_features:
        logger.warning(f"Insufficient features for correlation analysis. Using first {min_features} features.")
        top_indices = list(range(min(min_features, normal_data.shape[1])))
    
    # Extraer subconjunto de datos con las top características
    normal_subset = normal_data[:, top_indices]
    
    # Para anomalía, asegurar formato adecuado
    if anomaly_data.ndim == 1:
        anomaly_subset = anomaly_data[top_indices].reshape(1, -1)
    elif anomaly_data.shape[0] == 1:
        anomaly_subset = anomaly_data[:, top_indices]
    else:
        # Tomar promedio si hay múltiples anomalías
        anomaly_subset = np.mean(anomaly_data[:, top_indices], axis=0).reshape(1, -1)
    
    # Calcular matrices de correlación con manejo de errores
    try:
        normal_corr = np.corrcoef(normal_subset.T)
        # Ensure it's 2D
        if normal_corr.ndim < 2 or normal_corr.size == 1:
            normal_corr = np.array([[1.0]])  # Default for single feature
            if len(top_indices) > 1:
                normal_corr = np.eye(len(top_indices))  # Identity matrix for multiple features
            logger.warning("Could not compute meaningful normal correlation matrix. Using identity matrix.")
    except Exception as e:
        logger.warning(f"Error computing normal correlation matrix: {str(e)}")
        normal_corr = np.eye(len(top_indices))  # Default fallback
    
    # Para anomalía, si es un solo punto, no podemos calcular correlación directamente
    # Creamos datos sintéticos alrededor del punto anómalo
    try:
        if anomaly_subset.shape[0] < 5:
            # Crear perturbaciones pequeñas alrededor del punto anómalo
            n_synth = 20
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 0.1, (n_synth, len(top_indices)))
            anomaly_augmented = np.vstack([anomaly_subset] * n_synth) + noise
            anomaly_corr = np.corrcoef(anomaly_augmented.T)
        else:
            anomaly_corr = np.corrcoef(anomaly_subset.T)
        
        # Ensure it's 2D
        if anomaly_corr.ndim < 2 or anomaly_corr.size == 1:
            anomaly_corr = np.array([[1.0]])  # Default for single feature
            if len(top_indices) > 1:
                anomaly_corr = np.eye(len(top_indices))  # Identity matrix for multiple features
            logger.warning("Could not compute meaningful anomaly correlation matrix. Using identity matrix.")
    except Exception as e:
        logger.warning(f"Error computing anomaly correlation matrix: {str(e)}")
        anomaly_corr = np.eye(len(top_indices))  # Default fallback
    
    # Check that both matrices have same shape
    if normal_corr.shape != anomaly_corr.shape:
        logger.warning("Correlation matrices have different shapes. Adjusting...")
        max_size = max(normal_corr.shape[0], anomaly_corr.shape[0])
        
        # Create new matrices of the right size
        new_normal_corr = np.eye(max_size)
        new_anomaly_corr = np.eye(max_size)
        
        # Copy original data
        new_normal_corr[:normal_corr.shape[0], :normal_corr.shape[1]] = normal_corr
        new_anomaly_corr[:anomaly_corr.shape[0], :anomaly_corr.shape[1]] = anomaly_corr
        
        normal_corr = new_normal_corr
        anomaly_corr = new_anomaly_corr
    
    # Calcular diferencia entre matrices de correlación
    corr_diff = anomaly_corr - normal_corr
    
    # Crear nombres de características para visualización
    subset_names = []
    for i in top_indices[:normal_corr.shape[0]]:  # Limit to matrix size
        if feature_names and i < len(feature_names):
            subset_names.append(feature_names[i])
        else:
            subset_names.append(f"Feature_{i}")
    
    # Adjust if we don't have enough labels
    while len(subset_names) < normal_corr.shape[0]:
        subset_names.append(f"F_{len(subset_names)}")
    
    # Visualizar matrices de correlación
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Matriz de correlación normal
    sns.heatmap(normal_corr, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=subset_names, yticklabels=subset_names,
                ax=axes[0], square=True, cbar=False, vmin=-1, vmax=1)
    axes[0].set_title("Correlación en Datos Normales")
    
    # Matriz de correlación anómala
    sns.heatmap(anomaly_corr, annot=True, fmt=".2f", cmap="Reds", 
                xticklabels=subset_names, yticklabels=subset_names,
                ax=axes[1], square=True, cbar=False, vmin=-1, vmax=1)
    axes[1].set_title("Correlación en Datos Anómalos")
    
    # Diferencia de correlaciones
    sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=subset_names, yticklabels=subset_names,
                ax=axes[2], square=True, cbar=True, vmin=-2, vmax=2)
    axes[2].set_title("Diferencia de Correlación")
    
    plt.suptitle("Análisis de Correlación entre Características", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    return fig


class ContributionVisualizer:
    """
    Clase para generar visualizaciones avanzadas de contribuciones de características.
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Inicializa el visualizador.
        
        Args:
            feature_names: Nombres de las características
        """
        self.feature_names = feature_names
    
    def create_feature_contribution_report(self, 
                                         anomaly_data: np.ndarray, 
                                         normal_data: np.ndarray,
                                         detector: BaseAnomalyDetector,
                                         output_dir: str = "./") -> Dict[str, Any]:
        """
        Genera un informe completo con múltiples visualizaciones y análisis.
        
        Args:
            anomaly_data: Datos anómalos
            normal_data: Datos normales de referencia
            detector: Detector de anomalías
            output_dir: Directorio donde guardar visualizaciones
            
        Returns:
            Diccionario con resultados y rutas de archivos generados
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            "detector_type": type(detector).__name__,
            "anomaly_count": 1 if anomaly_data.ndim == 1 else anomaly_data.shape[0],
            "normal_count": normal_data.shape[0],
            "feature_count": anomaly_data.shape[1] if anomaly_data.ndim > 1 else len(anomaly_data),
            "visualizations": [],
            "top_features": []
        }
        
        # 1. Análisis de ensamble para identificar características importantes
        analyzer = EnsembleFeatureAnalyzer(detector, self.feature_names)
        results = analyzer.analyze(anomaly_data, normal_data)
        
        # Guardar top características
        report["top_features"] = results["ranked_features"][:10]
        
        # 2. Generar y guardar visualizaciones
        
        # 2.1 Comparación de métodos de análisis
        ensemble_fig = analyzer.plot_ensemble_comparison(results)
        ensemble_path = os.path.join(output_dir, "ensemble_comparison.png")
        ensemble_fig.savefig(ensemble_path)
        report["visualizations"].append({
            "name": "Ensemble Comparison",
            "path": ensemble_path,
            "description": "Comparación de diferentes métodos de análisis de características"
        })
        
        # 2.2 Comparación de distribuciones
        dist_fig = compare_feature_distributions(
            normal_data, anomaly_data, self.feature_names, top_n=6
        )
        dist_path = os.path.join(output_dir, "feature_distributions.png")
        dist_fig.savefig(dist_path)
        report["visualizations"].append({
            "name": "Feature Distributions",
            "path": dist_path,
            "description": "Comparación de distribuciones entre datos normales y anómalos"
        })
        
        # 2.3 Análisis de correlación
        corr_fig = visualize_feature_correlation_impact(
            anomaly_data, normal_data, self.feature_names, top_n=8
        )
        corr_path = os.path.join(output_dir, "correlation_analysis.png")
        corr_fig.savefig(corr_path)
        report["visualizations"].append({
            "name": "Correlation Analysis",
            "path": corr_path,
            "description": "Análisis de cambios en correlaciones entre características"
        })
        
        # 2.4 Si es un autoencoder, agregar visualización de reconstrucción
        if isinstance(detector, AutoencoderDetector):
            recon_analyzer = ReconstructionErrorAnalyzer(detector, self.feature_names)
            recon_fig = recon_analyzer.plot_reconstruction_comparison(
                anomaly_data, recon_analyzer.analyze(anomaly_data, normal_data)
            )
            recon_path = os.path.join(output_dir, "reconstruction_comparison.png")
            recon_fig.savefig(recon_path)
            report["visualizations"].append({
                "name": "Reconstruction Comparison",
                "path": recon_path,
                "description": "Comparación entre datos originales y reconstruidos por el autoencoder"
            })
        
        return report


if __name__ == "__main__":
    # Ejemplo de uso del análisis de contribución de características
    print("Iniciando demostración de análisis de contribución de características...")
    
    # 1. Generar datos sintéticos
    np.random.seed(42)
    
    # 1.1 Definir nombres de características para legibilidad
    feature_names = [
        "temperatura", "presión", "vibración", "voltaje", 
        "corriente", "frecuencia", "humedad", "ruido"
    ]
    n_features = len(feature_names)
    
    # 1.2 Generar datos normales
    n_normal = 1000
    normal_mean = np.array([100, 5.5, 0.5, 220, 10, 60, 45, 0.2])
    normal_cov = np.eye(n_features)
    # Añadir algunas correlaciones
    normal_cov[0, 4] = normal_cov[4, 0] = 0.7  # Temperatura y corriente
    normal_cov[1, 2] = normal_cov[2, 1] = 0.5  # Presión y vibración
    
    normal_data = np.random.multivariate_normal(normal_mean, normal_cov, n_normal)
    
    # 1.3 Generar algunos ejemplos anómalos
    # Anomalía 1: Alta temperatura y corriente
    anomaly1 = normal_mean.copy()
    anomaly1[0] = 150  # Alta temperatura
    anomaly1[4] = 18   # Alta corriente
    
    # Anomalía 2: Baja presión y alta vibración
    anomaly2 = normal_mean.copy()
    anomaly2[1] = 3.0  # Baja presión
    anomaly2[2] = 1.8  # Alta vibración
    
    # Anomalía 3: Múltiples parámetros anómalos
    anomaly3 = normal_mean.copy()
    anomaly3[0] = 130  # Alta temperatura
    anomaly3[3] = 190  # Bajo voltaje
    anomaly3[5] = 55   # Baja frecuencia
    anomaly3[6] = 90   # Alta humedad
    
    # Combinar anomalías
    anomalies = np.vstack([anomaly1, anomaly2, anomaly3])
    
    print(f"Datos generados: {n_normal} normales y {len(anomalies)} anómalos")
    
    # 2. Importar y configurar detectores de anomalías
    try:
        from models import IsolationForestDetector, AutoencoderDetector, LOFDetector
        
        print("\nCreando y entrenando detectores de anomalías...")
        
        # 2.1 Detector Isolation Forest
        iforest = IsolationForestDetector("IsolationForest")
        iforest.fit(normal_data)
        
        # 2.2 Detector Autoencoder
        autoencoder = AutoencoderDetector("Autoencoder")
        autoencoder.fit(normal_data)
        
        # 2.3 Detector LOF
        lof = LOFDetector("LOF")
        lof.fit(normal_data)
        
        print("Detectores entrenados correctamente")
        
        # 3. Análisis de contribución de características
        print("\nAnalizando contribuciones de características...")
        
        # 3.1 Crear directorio para resultados
        import os
        output_dir = "./feature_contribution_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 3.2 Analizar primera anomalía con diferentes detectores
        anomaly_to_analyze = anomalies[0:1]  # Primera anomalía
        
        # 3.3 Análisis con Isolation Forest
        print("\n3.1 Análisis con Isolation Forest")
        iforest_analyzer = EnsembleFeatureAnalyzer(iforest, feature_names)
        iforest_results = iforest_analyzer.analyze(anomaly_to_analyze, normal_data)
        
        print("Top 5 características contribuyentes:")
        for i, (feature, score) in enumerate(iforest_results["ranked_features"][:5]):
            print(f"  {i+1}. {feature}: {float(score):.4f}")
        
        # Visualizar y guardar
        iforest_fig = iforest_analyzer.plot_ensemble_comparison(iforest_results)
        iforest_fig.savefig(os.path.join(output_dir, "iforest_analysis.png"))
        print(f"Visualización guardada en {output_dir}/iforest_analysis.png")
        
        # 3.4 Análisis con Autoencoder
        print("\n3.2 Análisis con Autoencoder")
        autoencoder_analyzer = ReconstructionErrorAnalyzer(autoencoder, feature_names)
        autoencoder_results = autoencoder_analyzer.analyze(anomaly_to_analyze, normal_data)
        
        print("Top 5 características contribuyentes:")
        for i, (feature, score) in enumerate(autoencoder_results["ranked_features"][:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        # Visualizar y guardar reconstrucción
        recon_fig = autoencoder_analyzer.plot_reconstruction_comparison(anomaly_to_analyze, autoencoder_results)
        recon_fig.savefig(os.path.join(output_dir, "autoencoder_reconstruction.png"))
        print(f"Visualización guardada en {output_dir}/autoencoder_reconstruction.png")
        
        # 3.5 Análisis estadístico
        print("\n3.3 Análisis estadístico")
        stat_analyzer = StatisticalContributionAnalyzer(feature_names)
        stat_results = stat_analyzer.analyze(anomaly_to_analyze, normal_data, method="z_score")
        
        print("Top 5 características contribuyentes:")
        for i, (feature, score) in enumerate(stat_results["ranked_features"][:5]):
            print(f"  {i+1}. {feature}: {score}")
        
        # Visualizar distribuciones
        dist_fig = compare_feature_distributions(normal_data, anomaly_to_analyze, feature_names)
        dist_fig.savefig(os.path.join(output_dir, "feature_distributions.png"))
        print(f"Visualización guardada en {output_dir}/feature_distributions.png")
        
        # 3.6 Análisis de correlación
        print("\n3.4 Análisis de correlación")
        corr_fig = visualize_feature_correlation_impact(anomaly_to_analyze, normal_data, feature_names)
        corr_fig.savefig(os.path.join(output_dir, "correlation_analysis.png"))
        print(f"Visualización guardada en {output_dir}/correlation_analysis.png")
        
        # 3.7 Analizar todas las anomalías y comparar
        print("\n3.5 Comparativa de diferentes anomalías")
        
        # Crear tabla comparativa
        anomaly_comparison = []
        
        for i, anomaly in enumerate(anomalies):
            # Analizar con analizador de ensamble
            print(f"\nAnalizando anomalía {i+1}: {anomaly}")
            ensemble_analyzer = EnsembleFeatureAnalyzer(iforest, feature_names)
            single_anomaly = anomaly.reshape(1, -1)
            results = ensemble_analyzer.analyze(single_anomaly, normal_data)
            
            # Guardar características principales
            top_features = [name for name, _ in results["ranked_features"][:3]]
            anomaly_comparison.append({
                "anomaly_id": i+1,
                "top_features": top_features,
                "description": f"Anomalía {i+1}"
            })
            
            # Imprimir resultados
            print(f"Características principales para anomalía {i+1}:")
            for j, (feature, score) in enumerate(results["ranked_features"][:3]):
                print(f"  {j+1}. {feature}: {score:.4f}")
        
        # Generar informe comparativo
        print("\nComparación de anomalías:")
        for anomaly_info in anomaly_comparison:
            print(f"- {anomaly_info['description']}: {', '.join(anomaly_info['top_features'])}")
        
        # 4. Generar informe completo con ContributionVisualizer
        print("\n4. Generando informe completo de contribución de características")
        
        # 4.1 Crear visualizador y generar informe para la primera anomalía
        visualizer = ContributionVisualizer(feature_names=feature_names)
        report_dir = os.path.join(output_dir, "complete_report")
        
        report = visualizer.create_feature_contribution_report(
            anomaly_data=anomalies[0:1],
            normal_data=normal_data,
            detector=iforest,
            output_dir=report_dir
        )
        
        # 4.2 Imprimir resumen del informe
        print(f"\nInforme generado en: {report_dir}")
        print(f"Detector utilizado: {report['detector_type']}")
        print(f"Top 5 características contribuyentes:")
        for i, (feature, score) in enumerate(report['top_features'][:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        print("\nVisualizaciones generadas:")
        for viz in report["visualizations"]:
            print(f"- {viz['name']}: {viz['description']}")
            print(f"  Guardada en: {viz['path']}")
            
        # 5. Demostrar análisis de perturbación con visualización
        print("\n5. Demostración de análisis de perturbación")
        
        perturb_analyzer = PerturbationAnalyzer(iforest, feature_names)
        perturb_results = perturb_analyzer.analyze(anomalies[1:2], normal_data)
        
        print("Análisis de perturbación completado")
        print(f"Tiempo de análisis: {perturb_results['analysis_time']:.2f} segundos")
        print("Top 5 características identificadas por perturbación:")
        for i, (feature, score) in enumerate(perturb_results["ranked_features"][:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        # Visualizar resultados
        perturb_fig = perturb_analyzer.plot_feature_impact(perturb_results)
        perturb_path = os.path.join(output_dir, "perturbation_analysis.png")
        perturb_fig.savefig(perturb_path)
        print(f"Visualización guardada en {perturb_path}")
        
        # 6. Evaluación de rendimiento de los diferentes métodos de análisis
        print("\n6. Evaluación de rendimiento de métodos de análisis")
        
        # Tabla de tiempos de ejecución para diferentes tamaños de dataset
        print("\nTiempos de ejecución para análisis de anomalías:")
        print("{:<20} {:<15} {:<15} {:<15}".format("Método", "Anomalía única", "10 anomalías", "100 anomalías"))
        print("-" * 65)
        
        # Datos para prueba de rendimiento
        single_anomaly = anomalies[0:1]
        ten_anomalies = np.vstack([anomalies[0:1]] * 10)
        
        # Medir tiempos para diferentes métodos y tamaños
        for method_name, analyzer in [
            ("Estadístico", StatisticalContributionAnalyzer(feature_names)),
            ("Perturbación", PerturbationAnalyzer(iforest, feature_names)),
            ("Reconstrucción", ReconstructionErrorAnalyzer(autoencoder, feature_names)),
            ("Ensamble", EnsembleFeatureAnalyzer(iforest, feature_names))
        ]:
            # Medir para 1 anomalía
            start = time.time()
            if isinstance(analyzer, ReconstructionErrorAnalyzer):
                analyzer.analyze(single_anomaly, normal_data)
            else:
                analyzer.analyze(single_anomaly, normal_data)
            time_single = time.time() - start
            
            # Medir para 10 anomalías
            start = time.time()
            if isinstance(analyzer, ReconstructionErrorAnalyzer):
                analyzer.analyze(ten_anomalies, normal_data)
            else:
                analyzer.analyze(ten_anomalies, normal_data)
            time_ten = time.time() - start
            
            # Estimar tiempo para 100 anomalías
            time_hundred = time_ten * 10  # Estimación lineal
            
            print("{:<20} {:<15.4f}s {:<15.4f}s {:<15.4f}s".format(
                method_name, time_single, time_ten, time_hundred))
        
        print("\nDemostración completada exitosamente!")
        
    except ImportError as e:
        print(f"Error al importar detectores de anomalías: {e}")
        print("Asegúrese de que los modelos estén implementados correctamente.")
    
    except Exception as e:
        print(f"Error durante la demostración: {str(e)}")
        import traceback
        traceback.print_exc()