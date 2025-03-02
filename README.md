# Table Of Contents
1. [Table Of Contents](#table-of-contents)
2. [Sistema de Detección de Anomalías para Proyecciones Multidimensionales en Tiempo Real](#sistema-de-detección-de-anomalías-para-proyecciones-multidimensionales-en-tiempo-real)
   1. [Arquitectura general: Procesamiento en tiempo real combinado con capacidades de aprendizaje continuo:](#arquitectura-general-procesamiento-en-tiempo-real-combinado-con-capacidades-de-aprendizaje-continuo)
      1. [Capa de Ingesta y Preprocesamiento](#capa-de-ingesta-y-preprocesamiento)
      2. [Capa de Modelado y Detección](#capa-de-modelado-y-detección)
      3. [Capa de Adaptación y Aprendizaje](#capa-de-adaptación-y-aprendizaje)
      4. [Capa de Análisis y Visualización](#capa-de-análisis-y-visualización)
3. [Técnicas de modelado adecuadas](#técnicas-de-modelado-adecuadas)
   1. [Técnicas Basadas en Densidad](#técnicas-basadas-en-densidad)
   2. [Modelos de Aprendizaje no Supervisado](#modelos-de-aprendizaje-no-supervisado)
   3. [Modelos Específicos para Series Temporales](#modelos-específicos-para-series-temporales)
   4. [Enfoques de Ensamble](#enfoques-de-ensamble)
4. [Implementación de umbrales adaptativos](#implementación-de-umbrales-adaptativos)
   1. [Umbrales Basados en Estadísticas Móviles](#umbrales-basados-en-estadísticas-móviles)
   2. [Umbrales Probabilísticos](#umbrales-probabilísticos)
   3. [Umbrales Contextual-Adaptativos](#umbrales-contextual-adaptativos)
   4. [Sistema de Meta-Umbral](#sistema-de-meta-umbral)
5. [Mecanismos de adaptación automática](#mecanismos-de-adaptación-automática)
   1. [Ventanas Deslizantes con Pesos Decrecientes](#ventanas-deslizantes-con-pesos-decrecientes)
   2. [Detección de Deriva Conceptual](#detección-de-deriva-conceptual)
   3. [Aprendizaje Incremental](#aprendizaje-incremental)
   4. [Retroalimentación Semi-Supervisada](#retroalimentación-semi-supervisada)
6. [Equilibrio entre procesamiento en tiempo real y complejidad computacional](#equilibrio-entre-procesamiento-en-tiempo-real-y-complejidad-computacional)
   1. [Estratificación de Modelos](#estratificación-de-modelos)
   2. [Paralelización y Distribución](#paralelización-y-distribución)
7. [Estructura](#estructura)

# Sistema de Detección de Anomalías para Proyecciones Multidimensionales en Tiempo Real
El objetivo es diseñar un sistema robusto que pueda identificar comportamientos anómalos en datos multidimensionales proyectados en un plano 2D. 
## Arquitectura general: Procesamiento en tiempo real combinado con capacidades de aprendizaje continuo:
### Capa de Ingesta y Preprocesamiento
- **Colector de datos en tiempo real:** Interfaz para recibir el flujo de datos entrante
- **Normalización y transformación:** Preparación de datos para análisis posterior
- **Buffer de memoria:** Almacenamiento temporal para análisis de ventanas deslizantes
- **Detección de cambios de régimen:** Identificación de cambios en los modos de operación

### Capa de Modelado y Detección

- **Módulo de modelos múltiples:** Diferentes algoritmos de detección ejecutándose en paralelo
- **Subsistema de fusión:** Integración de resultados de varios modelos
- **Módulo de contextualización:** Incorporación de información contextual (hora, régimen, etc.)

### Capa de Adaptación y Aprendizaje

- **Módulo de reentrenamiento:** Actualización periódica de modelos con nuevos datos normales
- **Ajuste adaptativo de umbrales:** Modificación dinámica de límites de detección
- **Sistema de retroalimentación:** Incorporación de validaciones de expertos (opcional)

### Capa de Análisis y Visualización

- **Explicación de anomalías:** Identificación de variables contribuyentes
- **Panel de monitoreo:** Visualización en tiempo real de datos y alertas
- **Almacenamiento histórico:** Registro de eventos y anomalías para análisis posterior

# Técnicas de modelado adecuadas
Dada la naturaleza compleja y multidimensional de los datos, se usará un enfoque híbrido:
## Técnicas Basadas en Densidad
- **Local Outlier Factor (LOF):** Excelente para detectar anomalías locales, comparando la densidad de cada punto con sus vecinos. Útil cuando las anomalías no siguen patrones globales.
- **DBSCAN adaptativo:** Identificación de clusters en espacios de alta dimensionalidad con densidades variables.

## Modelos de Aprendizaje no Supervisado
- **Autoencoders profundos:** Muy efectivos para aprender representaciones comprimidas de datos normales. La reconstrucción de datos anómalos tendrá un error mayor.
- **Isolation Forest:** Eficiente para aislar observaciones anómalas mediante particiones aleatorias del espacio.
- **One-Class SVM:** Útil para delimitar el comportamiento normal en un espacio de características.

## Modelos Específicos para Series Temporales
- **HMM (Hidden Markov Models):** Capaces de modelar diferentes regímenes o estados del sistema.
- **LSTM autoencoders:** Efectivos para capturar dependencias temporales complejas y no lineales.

## Enfoques de Ensamble
- **Voting System:** Combinar resultados de varios detectores para mayor robustez.
- **Stacking:** Usar las predicciones de varios modelos como entrada para un meta-modelo.

# Implementación de umbrales adaptativos
Los umbrales estáticos no son adecuados para este tipo de sistema. Por tanto, mejor usar
## Umbrales Basados en Estadísticas Móviles
- Cálculo de medias y desviaciones estándar móviles sobre ventanas temporales
- Umbrales dinámicos basados en múltiplos de la desviación estándar (ej. μ ± kσ, k tipico entre 2 y 3)
- Actualización continua de estos parámetros con nuevos datos normales

## Umbrales Probabilísticos
- Modelado de la distribución de puntuaciones de anomalía usando técnicas como KDE (Kernel Density Estimation)
- Definición de umbrales basados en percentiles (ej. p99 como límite de comportamiento normal)
- Actualización periódica de estas distribuciones

## Umbrales Contextual-Adaptativos
- Umbrales diferentes para cada régimen o modo de operación identificado
- Matrices de transición para modelar los cambios aceptables entre estados consecutivos
- Factores de ajuste basados en variables contextuales (hora del día, carga de trabajo, etc.)

## Sistema de Meta-Umbral
- Monitoreo del rendimiento de los umbrales mismos
- Ajuste automático basado en tasas históricas de falsos positivos/negativos
- Implementación de lógica difusa para transiciones suaves entre umbrales

# Mecanismos de adaptación automática
Para que el sistema evolucione con cambios graduales en el comportamiento normal:
## Ventanas Deslizantes con Pesos Decrecientes
- Utilizar ventanas temporales donde los datos más recientes tienen mayor peso
- Aplicar una función de decaimiento exponencial para reducir gradualmente la influencia de datos antiguos
- Reentrenamiento periódico usando principalmente datos recientes

## Detección de Deriva Conceptual
- Implementar monitores de distribución que comparen datos recientes con históricos
- Utilizar técnicas como pruebas de Kolmogorov-Smirnov para detectar cambios estadísticamente significativos
- Activar reentrenamientos cuando se detecten cambios sustanciales en la distribución

## Aprendizaje Incremental
- Algoritmos que soporten actualización sin reentrenamiento completo
- Técnicas como _mini-batch gradient descent_ para autoencoders
- Utilizar modelos como Mondrian Forests que permiten actualizaciones incrementales

## Retroalimentación Semi-Supervisada
- Mecanismos para validar o rechazar anomalías detectadas
- Utilizar estas etiquetas para refinar continuamente los modelos
- Técnicas de active learning para solicitar feedback en casos dudosos

# Equilibrio entre procesamiento en tiempo real y complejidad computacional
## Estratificación de Modelos

- **Primera capa:** Modelos ligeros de alta velocidad (ej. algoritmos estadísticos simples)
- **Segunda capa:** Modelos de complejidad media activados solo cuando la primera capa detecta potenciales anomalías
- **Tercera capa:** Modelos complejos (ej. deep learning) que se ejecutan periódicamente o bajo demanda

## Paralelización y Distribución
- Frameworks de procesamiento distribuido como Apache Kafka o Spark Streaming
- Particionamiento de datos por ventanas temporales o conjuntos de características
- Ejecución paralela de diferentes modelos en múltiples nodos

------------

# Estructura
1. ```config.py``` - Configuración centralizada del sistema
2. ```preprocessing.py``` - Normalización y preparación de datos
3. ```models.py``` - Implementación de los diferentes modelos de detección
4. ```thresholds.py``` - Lógica para umbrales adaptativos
5. ```regime_detector.py``` - Detección de regímenes/modos de operación
6. ```feature_contribution.py``` - Análisis de contribución de variables
7. ```anomaly_detector.py``` - Clase principal integradora
8. ```main.py``` - Punto de entrada con ejemplos de uso
