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

DISCHARGE_DATA_PATH = 'C:\\Users\\Ruben\\Documents\\python-disruption-predictor\\remuestreados' # Path to the directory containing discharge data files
DISCHARGES_ETIQUETATION_DATA_FILE = '..\\rust-disruptor-predictor\\discharges-c23.txt' # Path to the file containing discharge etiquetation data, which includes the discharge and the time when the disruption happened, or 0 if it didn't happen.


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
        
    def load_discharge_list(self):
        """Load the list of discharges and their labels."""
        discharge_df = pd.read_csv(self.discharge_list_path, sep='\\s+', header=None, 
                                   names=['discharge_id', 'disruption_time'])
        
        for _, row in discharge_df.iterrows():
            discharge_id = str(int(row['discharge_id']))
            disruption_time = row['disruption_time']
            # Label: 1 if disruptive (non-zero disruption time), 0 if not
            label = 1 if disruption_time > 0 else 0
            self.discharge_labels[discharge_id] = label
        
        return discharge_df
    
    def load_feature_data(self, discharge_id, feature_num):
        """Load a specific feature file for a discharge."""
        # Find the appropriate file
        feature_str = f"{int(feature_num):02d}"
        feature_files = [f for f in os.listdir(self.data_path) 
                        if f.startswith(f"DES_{discharge_id}_{feature_str}")]
        
        if not feature_files:
            print(f"No file found for discharge {discharge_id}, feature {feature_num}")
            return None
        
        # Load the feature data
        feature_path = os.path.join(self.data_path, feature_files[0])
        feature_data = pd.read_csv(feature_path, sep='\\s+', header=None, 
                                  names=['time', 'value'])
        return feature_data
    
    def load_all_data(self):
        """Load all discharge data for all features."""
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
        """Extract features from the time series data."""
        X = []
        y = []
        discharge_ids = []
        
        for discharge_id, features in self.discharge_data.items():
            # Extract relevant features from each time series
            feature_vector = []
            
            for feature_num, feature_data in features.items():
                # Simple features: mean, std, min, max, etc.
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
            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, discharge_ids
    
    def prepare_lstm_data(self, X_scaled, y, sequence_length=10):
        """Prepare data specifically for LSTM model."""
        # Reshape data for LSTM: [samples, time steps, features]
        X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        return X_lstm, y
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train all models and evaluate them."""
        X_scaled, y, discharge_ids = self.prepare_features()
        
        # Split data ensuring test set has at least one of each class
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X_scaled, y, discharge_ids, test_size=test_size, random_state=random_state,
            stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train OCSVM - train only on non-disruptive cases (label 0)
        X_train_normal = X_train[y_train == 0]
        self.ocsvm_model = OneClassSVM(kernel='rbf', gamma='auto')
        self.ocsvm_model.fit(X_train_normal)
        
        # Train Isolation Forest - again, train only on normal data
        self.iforest_model = IsolationForest(random_state=random_state, contamination=0.1)
        self.iforest_model.fit(X_train_normal)
        
        # Train LSTM
        X_lstm_train, _ = self.prepare_lstm_data(X_train, y_train)
        X_lstm_test, _ = self.prepare_lstm_data(X_test, y_test)
        
        # LSTM model for binary classification
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model.fit(X_lstm_train, y_train, epochs=100, batch_size=16, verbose=1, 
                          validation_split=0.2)
        
        # Evaluate each model individually
        ocsvm_predictions = self.predict_ocsvm(X_test)
        iforest_predictions = self.predict_iforest(X_test)
        lstm_predictions = self.predict_lstm(X_lstm_test)
        
        print("\nOCSVM Results:")
        print(classification_report(y_test, ocsvm_predictions))
        
        print("\nIsolation Forest Results:")
        print(classification_report(y_test, iforest_predictions))
        
        print("\nLSTM Results:")
        print(classification_report(y_test, lstm_predictions))
        
        # Evaluate combined model
        combined_predictions = self.combine_predictions(ocsvm_predictions, iforest_predictions, lstm_predictions)
        
        print("\nCombined Model Results:")
        print(classification_report(y_test, combined_predictions))
        
        # Save test set details for later analysis
        self.test_data = {
            'X_test': X_test,
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
        """Predict using OCSVM model."""
        # OCSVM returns: 1 for normal, -1 for anomaly
        # We want: 0 for normal, 1 for anomaly/disruption
        raw_pred = self.ocsvm_model.predict(X)
        return (raw_pred == -1).astype(int)  # Convert to binary labels
    
    def predict_iforest(self, X):
        """Predict using Isolation Forest."""
        # Isolation Forest returns: 1 for normal, -1 for anomaly
        # We want: 0 for normal, 1 for anomaly/disruption
        raw_pred = self.iforest_model.predict(X)
        return (raw_pred == -1).astype(int)
    
    def predict_lstm(self, X):
        """Predict using LSTM model."""
        raw_pred = self.lstm_model.predict(X)
        return (raw_pred > 0.5).astype(int).flatten()
    
    def combine_predictions(self, ocsvm_pred, iforest_pred, lstm_pred):
        """Combine predictions using majority voting."""
        # Stack predictions
        all_preds = np.vstack([ocsvm_pred, iforest_pred, lstm_pred])
        # Count votes for class 1 (disruption)
        votes = np.sum(all_preds, axis=0)
        # If majority votes for disruption (class 1), predict 1
        return (votes >= 2).astype(int)
    
    def predict_discharge(self, discharge_id):
        """Predict for a specific discharge and show detailed voting results."""
        if discharge_id not in self.discharge_data:
            print(f"Discharge {discharge_id} not found in data")
            return None
            
        # Extract features for this discharge
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
            
        # Flatten features list
        feature_vector = []
        for i in range(1, 8):
            if i in features:
                feature_vector.extend(features[i])
            else:
                # Fill with zeros for missing features
                feature_vector.extend([0, 0, 0, 0, 0, 0])
                
        # Scale the feature vector
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Get predictions from each model
        ocsvm_pred = self.predict_ocsvm(X_scaled)[0]
        iforest_pred = self.predict_iforest(X_scaled)[0]
        lstm_pred = self.predict_lstm(X_lstm)[0]
        
        # Final prediction using majority voting
        final_pred = 1 if (ocsvm_pred + iforest_pred + lstm_pred >= 2) else 0
        
        # Print results
        print(f"\nPrediction results for discharge {discharge_id}:")
        print(f"OCSVM: {'Disruptive' if ocsvm_pred == 1 else 'Non-disruptive'}")
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
        """Analyze and print detailed results for test set."""
        if not hasattr(self, 'test_data'):
            print("No test data available. Run train_models first.")
            return
            
        X_test = self.test_data['X_test']
        X_lstm_test = self.test_data['X_lstm_test']
        y_test = self.test_data['y_test']
        ids_test = self.test_data['ids_test']
        
        ocsvm_pred = self.predict_ocsvm(X_test)
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
            
        # Calculate overall metrics
        print("\nOverall Accuracy:")
        print(f"OCSVM: {accuracy_score(y_test, ocsvm_pred):.4f}")
        print(f"Isolation Forest: {accuracy_score(y_test, iforest_pred):.4f}")
        print(f"LSTM: {accuracy_score(y_test, lstm_pred):.4f}")
        print(f"Combined: {accuracy_score(y_test, combined_pred):.4f}")

# Main code to run the classifier
if __name__ == "__main__":
    # Paths to data
    data_path = DISCHARGE_DATA_PATH
    discharge_list_path = DISCHARGES_ETIQUETATION_DATA_FILE
    
    # Create and run the classifier
    classifier = DisruptionClassifier(data_path, discharge_list_path)
    classifier.load_all_data()
    
    # Train all models
    results = classifier.train_models(test_size=0.3)
    
    # Analyze test set in detail
    classifier.analyze_test_set()
    
    print("\nPredicting for specific discharges:")
    for discharge_id in list(classifier.discharge_labels.keys()):
        classifier.predict_discharge(discharge_id)

    # Print summary of results
    print("\nSummary of results:")
    for discharge_id, label in classifier.discharge_labels.items():
        print(f"Discharge {discharge_id}: classified as {'Disruptive' if label == 1 else 'Non-disruptive'}, actual label: {'Disruptive' if label == 1 else 'Non-disruptive'}")
    print("\nAll models trained and evaluated successfully.")

    # Print precision and recall for each model
    print("\nPrecision and Recall:")
    for model_name, predictions in results.items():
        if model_name != 'true':
            precision = np.sum((predictions == 1) & (results['true'] == 1)) / np.sum(predictions == 1)
            recall = np.sum((predictions == 1) & (results['true'] == 1)) / np.sum(results['true'] == 1)
            print(f"{model_name}: Precision = {precision:.4f}, Recall = {recall:.4f}")