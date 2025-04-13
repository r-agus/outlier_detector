import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from disruption_classifier.ocsvm_data_processor import process_data_for_classifier

DISCHARGE_DATA_PATH = 'C:\\Users\\Ruben\\Documents\\python-disruption-predictor\\remuestreados'
DISCHARGES_ETIQUETATION_DATA_FILE = 'C:\\Users\Ruben\\Documents\\tfg_julian\\rust-disruptor-predictor\\discharges-c23.txt'


class DisruptionClassifier:
    def __init__(self, data_path, discharge_list_path):
        self.data_path = data_path
        self.discharge_list_path = discharge_list_path
        self.discharge_data = {}
        self.discharge_labels = {}
        self.ocsvm_model = None
        self.iforest_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.ocsvm_scaler = StandardScaler()

    def load_discharge_list(self):
        discharge_df = pd.read_csv(self.discharge_list_path, sep='\\s+', header=None,
                                   names=['discharge_id', 'disruption_time'])

        for _, row in discharge_df.iterrows():
            discharge_id = str(int(row['discharge_id']))
            disruption_time = row['disruption_time']
            label = 1 if disruption_time > 0 else 0
            self.discharge_labels[discharge_id] = label

        return discharge_df

    def load_feature_data(self, discharge_id, feature_num):
        feature_str = f"{int(feature_num):02d}"
        feature_files = [f for f in os.listdir(self.data_path)
                         if f.startswith(f"DES_{discharge_id}_{feature_str}")]

        if not feature_files:
            print(f"No file found for discharge {discharge_id}, feature {feature_num}")
            return None

        feature_path = os.path.join(self.data_path, feature_files[0])
        feature_data = pd.read_csv(feature_path, sep='\\s+', header=None,
                                   names=['time', 'value'])
        return feature_data

    def load_all_data(self):
        discharge_df = self.load_discharge_list()

        for _, row in discharge_df.iterrows():
            discharge_id = str(int(row['discharge_id']))
            features = {}

            for feature_num in range(1, 8):
                feature_data = self.load_feature_data(discharge_id, feature_num)
                if feature_data is not None:
                    features[feature_num] = feature_data

            if features:
                self.discharge_data[discharge_id] = features

        print(f"Loaded data for {len(self.discharge_data)} discharges")

    def prepare_features(self):
        X = []
        y = []
        discharge_ids = []

        for discharge_id, features in self.discharge_data.items():
            feature_vector = []

            for feature_num, feature_data in features.items():
                feature_vector.extend([
                    feature_data['value'].mean(),
                    feature_data['value'].std(),
                    feature_data['value'].min(),
                    feature_data['value'].max(),
                    feature_data['value'].skew(),
                    feature_data['value'].kurtosis()
                ])

            X.append(feature_vector)
            y.append(self.discharge_labels[discharge_id])
            discharge_ids.append(discharge_id)

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, discharge_ids

    def prepare_ocsvm_features(self):
        print("Extracting window-based features for OCSVM...")

        discharge_ids = list(self.discharge_labels.keys())

        discharge_features = process_data_for_classifier(
            self.data_path, discharge_ids, range(1, 8), window_size=16, debug_output=".\\debug_output"
        )

        X = []
        y = []
        processed_discharge_ids = []

        for discharge_id, features in discharge_features.items():
            if discharge_id in self.discharge_labels:
                X.append(features)
                y.append(self.discharge_labels[discharge_id])
                processed_discharge_ids.append(discharge_id)

        X = np.array(X)
        y = np.array(y)

        print(f"Processed {len(processed_discharge_ids)} discharges with {X.shape[1]} features for OCSVM")

        X_scaled = self.ocsvm_scaler.fit_transform(X)

        return X_scaled, y, processed_discharge_ids

    def prepare_lstm_data(self, X_scaled, y, sequence_length=10):
        X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        return X_lstm, y

    def train_models(self, test_size=0.2, random_state=42, ocsvm_params=None, iforest_params=None, lstm_params=None):
        X_scaled, y, discharge_ids = self.prepare_features()
        X_ocsvm_scaled, y_ocsvm, discharge_ids_ocsvm = self.prepare_ocsvm_features()

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X_scaled, y, discharge_ids, test_size=test_size, random_state=random_state,
            stratify=y
        )

        X_ocsvm_train, X_ocsvm_test, y_ocsvm_train, y_ocsvm_test = train_test_split(
            X_ocsvm_scaled, y_ocsvm, test_size=test_size, random_state=random_state,
            stratify=y_ocsvm
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        X_ocsvm_train_normal = X_ocsvm_train[y_ocsvm_train == 0]

        if ocsvm_params is None:
            ocsvm_params = {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'auto'}

        print("\n" + "=" * 50)
        print("OCSVM Parameters (using window-based features):")
        print(f"  - kernel: {ocsvm_params['kernel']}")
        print(f"  - nu: {ocsvm_params['nu']}")
        print(f"  - gamma: {ocsvm_params['gamma']}")
        print("=" * 50)

        self.ocsvm_model = OneClassSVM(
            kernel=ocsvm_params['kernel'],
            nu=ocsvm_params['nu'],
            gamma=ocsvm_params['gamma']
        )
        self.ocsvm_model.fit(X_ocsvm_train_normal)

        X_train_normal = X_train[y_train == 0]

        if iforest_params is None:
            iforest_params = {'n_estimators': 100, 'contamination': 0.1, 'max_features': 1.0}

        print("\n" + "=" * 50)
        print("Isolation Forest Parameters (using standard features):")
        print(f"  - n_estimators: {iforest_params['n_estimators']}")
        print(f"  - contamination: {iforest_params['contamination']}")
        print(f"  - max_features: {iforest_params['max_features']}")
        print("=" * 50)

        self.iforest_model = IsolationForest(
            random_state=random_state,
            n_estimators=int(iforest_params['n_estimators']),
            contamination=float(iforest_params['contamination']),
            max_features=float(iforest_params['max_features'])
        )
        self.iforest_model.fit(X_train_normal)

        X_lstm_train, _ = self.prepare_lstm_data(X_train, y_train)
        X_lstm_test, _ = self.prepare_lstm_data(X_test, y_test)

        if lstm_params is None:
            lstm_params = {
                'layer1_units': 64,
                'layer2_units': 32,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 100
            }

        print("\n" + "=" * 50)
        print("LSTM Network Parameters:")
        print(f"  - layer1_units: {lstm_params['layer1_units']}")
        print(f"  - layer2_units: {lstm_params['layer2_units']}")
        print(f"  - dropout_rate: {lstm_params['dropout_rate']}")
        print(f"  - learning_rate: {lstm_params['learning_rate']}")
        print(f"  - batch_size: {lstm_params['batch_size']}")
        print(f"  - epochs: {lstm_params['epochs']}")
        print("=" * 50 + "\n")

        self.lstm_model = Sequential([
            LSTM(lstm_params['layer1_units'], input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(lstm_params['dropout_rate']),
            LSTM(lstm_params['layer2_units']),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lstm_params['learning_rate'])

        self.lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model.fit(
            X_lstm_train, y_train,
            epochs=lstm_params['epochs'],
            batch_size=lstm_params['batch_size'],
            verbose=1,
            validation_split=0.2
        )

        ocsvm_predictions = self.predict_ocsvm(X_ocsvm_test)
        iforest_predictions = self.predict_iforest(X_test)
        lstm_predictions = self.predict_lstm(X_lstm_test)

        print("\nOCSVM Results (window-based features):")
        print(classification_report(y_ocsvm_test, ocsvm_predictions))

        print("\nIsolation Forest Results (standard features):")
        print(classification_report(y_test, iforest_predictions))

        print("\nLSTM Results (standard features):")
        print(classification_report(y_test, lstm_predictions))

        combined_predictions = self.combine_predictions(
            self.predict_ocsvm(self.ocsvm_scaler.transform(X_test)),
            iforest_predictions,
            lstm_predictions
        )

        print("\nCombined Model Results:")
        print(classification_report(y_test, combined_predictions))

        self.test_data = {
            'X_test': X_test,
            'X_ocsvm_test': X_ocsvm_test,
            'y_test': y_test,
            'ids_test': ids_test,
            'X_lstm_test': X_lstm_test
        }

        return {
            'ocsvm': ocsvm_predictions,
            'iforest': iforest_predictions,
            'lstm': lstm_predictions,
            'combined': combined_predictions,
            'true': y_test,
            'ids': ids_test
        }

    def predict_ocsvm(self, X):
        raw_pred = self.ocsvm_model.predict(X)
        return (raw_pred == -1).astype(int)

    def predict_iforest(self, X):
        raw_pred = self.iforest_model.predict(X)
        return (raw_pred == -1).astype(int)

    def predict_lstm(self, X):
        raw_pred = self.lstm_model.predict(X)
        return (raw_pred > 0.5).astype(int).flatten()

    def combine_predictions(self, ocsvm_pred, iforest_pred, lstm_pred):
        all_preds = np.vstack([ocsvm_pred, iforest_pred, lstm_pred])
        votes = np.sum(all_preds, axis=0)
        return (votes >= 2).astype(int)

    def predict_discharge(self, discharge_id):
        if discharge_id not in self.discharge_data:
            print(f"Discharge {discharge_id} not found in data")
            return None

        features = {}
        for feature_num, feature_data in self.discharge_data[discharge_id].items():
            features[feature_num] = [
                feature_data['value'].mean(),
                feature_data['value'].std(),
                feature_data['value'].min(),
                feature_data['value'].max(),
                feature_data['value'].skew(),
                feature_data['value'].kurtosis()
            ]

        feature_vector = []
        for i in range(1, 8):
            if i in features:
                feature_vector.extend(features[i])
            else:
                feature_vector.extend([0, 0, 0, 0, 0, 0])

        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

        discharge_features = process_data_for_classifier(
            self.data_path, [discharge_id], range(1, 8), window_size=16
        )

        if discharge_id in discharge_features:
            X_ocsvm = np.array([discharge_features[discharge_id]])
            X_ocsvm_scaled = self.ocsvm_scaler.transform(X_ocsvm)
        else:
            print(f"Warning: Could not extract advanced features for discharge {discharge_id}. Using standard features for OCSVM.")
            X_ocsvm_scaled = X_scaled

        ocsvm_pred = self.predict_ocsvm(X_ocsvm_scaled)[0]
        iforest_pred = self.predict_iforest(X_scaled)[0]
        lstm_pred = self.predict_lstm(X_lstm)[0]

        final_pred = 1 if (ocsvm_pred + iforest_pred + lstm_pred >= 2) else 0

        print(f"\nPrediction results for discharge {discharge_id}:")
        print(f"OCSVM (window features): {'Disruptive' if ocsvm_pred == 1 else 'Non-disruptive'}")
        print(f"Isolation Forest: {'Disruptive' if iforest_pred == 1 else 'Non-disruptive'}")
        print(f"LSTM: {'Disruptive' if lstm_pred == 1 else 'Non-disruptive'}")
        print(f"Final prediction: {'Disruptive' if final_pred == 1 else 'Non-disruptive'}")
        print(f"Actual label: {'Disruptive' if self.discharge_labels[discharge_id] == 1 else 'Non-disruptive'}")

        return {
            'ocsvm': ocsvm_pred,
            'iforest': iforest_pred,
            'lstm': lstm_pred,
            'final': final_pred,
            'actual': self.discharge_labels[discharge_id]
        }

    def analyze_test_set(self):
        if not hasattr(self, 'test_data'):
            print("No test data available. Run train_models first.")
            return

        X_test = self.test_data['X_test']
        X_ocsvm_test = self.test_data['X_ocsvm_test']
        X_lstm_test = self.test_data['X_lstm_test']
        y_test = self.test_data['y_test']
        ids_test = self.test_data['ids_test']

        ocsvm_pred = self.predict_ocsvm(X_ocsvm_test)
        iforest_pred = self.predict_iforest(X_test)
        lstm_pred = self.predict_lstm(X_lstm_test)
        combined_pred = self.combine_predictions(ocsvm_pred, iforest_pred, lstm_pred)

        print("\nDetailed test set results:")
        for i, discharge_id in enumerate(ids_test):
            print(f"\nDischarge {discharge_id}:")
            print(f"Actual: {'Disruptive' if y_test[i] == 1 else 'Non-disruptive'}")
            print(f"OCSVM: {'Disruptive' if ocsvm_pred[i] == 1 else 'Non-disruptive'}")
            print(f"IForest: {'Disruptive' if iforest_pred[i] == 1 else 'Non-disruptive'}")
            print(f"LSTM: {'Disruptive' if lstm_pred[i] == 1 else 'Non-disruptive'}")
            print(f"Combined: {'Disruptive' if combined_pred[i] == 1 else 'Non-disruptive'}")

        print("\nOverall Accuracy:")
        print(f"OCSVM: {accuracy_score(y_test, ocsvm_pred):.4f}")
        print(f"Isolation Forest: {accuracy_score(y_test, iforest_pred):.4f}")
        print(f"LSTM: {accuracy_score(y_test, lstm_pred):.4f}")
        print(f"Combined: {accuracy_score(y_test, combined_pred):.4f}")


if __name__ == "__main__":
    data_path = DISCHARGE_DATA_PATH
    discharge_list_path = DISCHARGES_ETIQUETATION_DATA_FILE

    classifier = DisruptionClassifier(data_path, discharge_list_path)
    classifier.load_all_data()

    results = classifier.train_models(test_size=0.3)

    classifier.analyze_test_set()

    print("\nPredicting for specific discharges:")
    sample_discharges = list(classifier.discharge_labels.keys())[:5]
    for discharge_id in sample_discharges:
        classifier.predict_discharge(discharge_id)

    print("\nAll models trained and evaluated successfully.")

    print("\nPrecision and Recall:")
    for model_name, predictions in results.items():
        if model_name != 'true' and model_name != 'ids':
            precision = np.sum((predictions == 1) & (results['true'] == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
            recall = np.sum((predictions == 1) & (results['true'] == 1)) / np.sum(results['true'] == 1) if np.sum(results['true'] == 1) > 0 else 0
            print(f"{model_name}: Precision = {precision:.4f}, Recall = {recall:.4f}")