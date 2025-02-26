# -*- coding: utf-8 -*-
"""Compare all detection algorithms by plotting decision boundaries and
the number of decision boundaries.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.metrics import classification_report, confusion_matrix

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD

from pyod.models.dif import DIF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.suod import SUOD
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.kpca import KPCA
from pyod.models.lunar import LUNAR

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define the number of inliers and outliers
n_samples = 500
outliers_fraction = 0.2
clusters_separation = [0, 10]

# Compare given detectors under given settings
# Initialize the data
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
				 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
				 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
				 LOF(n_neighbors=50)]

# Show the statics of the data
print('Number of inliers: %i' % n_inliers)
print('Number of outliers: %i' % n_outliers)
print(
	'Ground truth shape is {shape}. Outlier are 1 and inliers are 0.\n'.format(
		shape=ground_truth.shape))
print(ground_truth, '\n')

random_state = 42
# Define nine outlier detection tools to be compared
classifiers = {
	'Angle-based Outlier Detector (ABOD)':
		ABOD(contamination=outliers_fraction),
	'K Nearest Neighbors (KNN)': KNN(
		contamination=outliers_fraction),
	'Average KNN': KNN(method='mean',
					   contamination=outliers_fraction),
	'Median KNN': KNN(method='median',
					  contamination=outliers_fraction),
	'Local Outlier Factor (LOF)':
		LOF(n_neighbors=35, contamination=outliers_fraction),

	'Isolation Forest': IForest(contamination=outliers_fraction,
								random_state=random_state),
	'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction,
									   random_state=random_state),
	'INNE': INNE(
		max_samples=2, contamination=outliers_fraction,
		random_state=random_state,
	),

	'Locally Selective Combination (LSCP)': LSCP(
		detector_list, contamination=outliers_fraction,
		random_state=random_state),
	'Feature Bagging':
		FeatureBagging(LOF(n_neighbors=35),
					   contamination=outliers_fraction,
					   random_state=random_state),
	'SUOD': SUOD(contamination=outliers_fraction),

	'Minimum Covariance Determinant (MCD)': MCD(
		contamination=outliers_fraction, random_state=random_state),

	'Principal Component Analysis (PCA)': PCA(
		contamination=outliers_fraction, random_state=random_state),
	'KPCA': KPCA(
		contamination=outliers_fraction),

	'Probabilistic Mixture Modeling (GMM)': GMM(contamination=outliers_fraction,
												random_state=random_state),

	'LMDD': LMDD(contamination=outliers_fraction,
				 random_state=random_state),

	'Histogram-based Outlier Detection (HBOS)': HBOS(
		contamination=outliers_fraction),

	'Copula-base Outlier Detection (COPOD)': COPOD(
		contamination=outliers_fraction),

	'ECDF-baseD Outlier Detection (ECOD)': ECOD(
		contamination=outliers_fraction),
	'Kernel Density Functions (KDE)': KDE(contamination=outliers_fraction),

	'QMCD': QMCD(
		contamination=outliers_fraction),

	'Sampling': Sampling(
		contamination=outliers_fraction),

	'LUNAR': LUNAR(),

	'Cluster-based Local Outlier Factor (CBLOF)':
		CBLOF(contamination=outliers_fraction,
			  check_estimator=False, random_state=random_state),

	'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
}

# Show all detectors
for i, clf in enumerate(classifiers.keys()):
	print('Model', i + 1, clf)

# Fit the models with the generated data and
# compare model performances
for num_it, offset in enumerate(clusters_separation):
	np.random.seed(42)
	# Data generation
	X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
	X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
	X = np.r_[X1, X2]
	# Add outliers
	X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

	# Fit the model
	plt.figure(figsize=(20, 22))
	for i, (clf_name, clf) in enumerate(classifiers.items()):
		print()
		print(i + 1, 'fitting', clf_name)
		# fit the data and tag outliers
		clf.fit(X)
		scores_pred = clf.decision_function(X) * -1
		y_pred = clf.predict(X)
		threshold = percentile(scores_pred, 100 * outliers_fraction)
		
		# Calcular métricas detalladas
		print(f"\n{i+1}. {clf_name}")
		print(classification_report(ground_truth, y_pred, target_names=["inlier", "outlier"]))
		print("Matriz de confusión:")
		print(confusion_matrix(ground_truth, y_pred))

		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
		Z = Z.reshape(xx.shape)
		subplot = plt.subplot(5, 5, i + 1)
		if threshold > Z.max():
			levels_inliers = np.linspace(Z.max(), threshold, 7)
		else:
			levels_inliers = np.linspace(threshold, Z.max(), 7)

		if threshold > Z.min():
			levels_outliers = np.linspace(Z.min(), threshold, 7)
		else:
			levels_outliers = np.linspace(threshold, Z.min(), 7)
		
		# Rellenar áreas
		cs_outliers = subplot.contourf(xx, yy, Z, levels=levels_outliers, 
								cmap=plt.cm.Blues_r, alpha=0.7)
		cs_inliers = subplot.contourf(xx, yy, Z, levels=levels_inliers, 
								cmap=plt.cm.Oranges_r, alpha=0.7)
		
		# Línea roja para el umbral
		# subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
		
		# Puntos y leyenda
		b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
					  s=20, edgecolor='k', label='Inliers reales')
		c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
					  s=20, edgecolor='k', label='Outliers reales')
		
		# Barra de color
		plt.colorbar(cs_inliers, ax=subplot, label='Puntuación de inlier')
		subplot.set_title(f"{i+1}. {clf_name}\nErrores: {(y_pred != ground_truth).sum()}", 
					fontsize=9)
		
		subplot.legend(loc='upper right', fontsize=6)

		subplot.set_xlim((-7, 7))
		subplot.set_ylim((-7, 7))
	plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
	plt.suptitle("Comparación de algoritmos de detección de outliers (naranja=inlier, azul=outlier)",
			y=0.95, fontsize=20)
	plt.savefig(f"ALL - {n_samples} samples - {n_samples * outliers_fraction} outliers - {num_it}.png", 
			dpi=300, bbox_inches='tight')
plt.show()