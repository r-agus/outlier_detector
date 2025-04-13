import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import model-specific modules
from ocsvm_model import prepare_features_for_ocsvm, train_ocsvm_model, predict_with_ocsvm, prepare_ocsvm_features_for_single_discharge
from iforest_model import prepare_features_for_iforest, train_iforest_model, predict_with_iforest, prepare_iforest_features_for_single_discharge
from lstm_model import prepare_features_for_lstm, train_lstm_model, predict_with_lstm, prepare_lstm_features_for_single_discharge

DISCHARGE_DATA_PATH = 'C:\\Users\\Ruben\\Documents\\python-disruption-predictor\\remuestreados' # Path to the directory containing discharge data files
DISCHARGES_ETIQUETATION_DATA_FILE = 'C:\\Users\\Ruben\\Documents\\tfg_julian\\rust-disruptor-predictor\\discharges-c23.txt' # Path to the file containing discharge etiquetation data, which includes the discharge and the time when the disruption happened, or 0 if it didn't happen.

class DisruptionClassifier:
    def __init__(self, data_path, discharge_list_path):
        self.data_path = data_path
        self.discharge_list_path = discharge_list_path
        self.discharge_data = {}
        self.discharge_labels = {}
        self.ocsvm_model = None
        self.iforest_model = None
        self.lstm_model = None
        self.iforest_scaler = None
        self.lstm_scaler = None
        self.advanced_features = None  # To store advanced features for OCSVM
        
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
    
    def train_models(self, test_size=0.2, random_state=42, ocsvm_params=None, iforest_params=None, lstm_params=None):
        """Train all models and evaluate them."""
        # Prepare features for OCSVM (including advanced feature extraction)
        X_ocsvm, y_ocsvm, ids_ocsvm, self.advanced_features = prepare_features_for_ocsvm(self.discharge_data, self.discharge_labels)
        
        # Prepare features for Isolation Forest
        X_iforest, y_iforest, ids_iforest, self.iforest_scaler = prepare_features_for_iforest(self.discharge_data, self.discharge_labels)
        
        # Prepare features for LSTM
        X_lstm, y_lstm, ids_lstm, self.lstm_scaler = prepare_features_for_lstm(self.discharge_data, self.discharge_labels)
        
        # Split data for each model
        X_ocsvm_train, X_ocsvm_test, y_ocsvm_train, y_ocsvm_test, ids_ocsvm_train, ids_ocsvm_test = train_test_split(
            X_ocsvm, y_ocsvm, ids_ocsvm, test_size=test_size, random_state=random_state,
            stratify=y_ocsvm
        )
        
        X_iforest_train, X_iforest_test, y_iforest_train, y_iforest_test, ids_iforest_train, ids_iforest_test = train_test_split(
            X_iforest, y_iforest, ids_iforest, test_size=test_size, random_state=random_state,
            stratify=y_iforest
        )
        
        X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test, ids_lstm_train, ids_lstm_test = train_test_split(
            X_lstm, y_lstm, ids_lstm, test_size=test_size, random_state=random_state,
            stratify=y_lstm
        )
        
        print(f"Training set: {len(X_ocsvm_train)} samples")
        print(f"Test set: {len(X_ocsvm_test)} samples")
        print(f"Test size: {test_size}")
        
        # Train models
        self.ocsvm_model = train_ocsvm_model(X_ocsvm_train, y_ocsvm_train, ocsvm_params)
        self.iforest_model = train_iforest_model(X_iforest_train, y_iforest_train, iforest_params, random_state)
        self.lstm_model = train_lstm_model(X_lstm_train, y_lstm_train, lstm_params)
        
        # Evaluate each model individually
        ocsvm_predictions = self.predict_ocsvm(X_ocsvm_test)
        iforest_predictions = self.predict_iforest(X_iforest_test)
        lstm_predictions = self.predict_lstm(X_lstm_test)
        
        print("\nOCSVM Results:")
        print(classification_report(y_ocsvm_test, ocsvm_predictions))
        
        print("\nIsolation Forest Results:")
        print(classification_report(y_iforest_test, iforest_predictions))
        
        print("\nLSTM Results:")
        print(classification_report(y_lstm_test, lstm_predictions))
        
        # To combine predictions, we need to ensure they're aligned by discharge ID
        # Create alignment between ids_*_test lists
        all_ids = list(set(ids_ocsvm_test) & set(ids_iforest_test) & set(ids_lstm_test))
        
        # Create aligned predictions and labels
        aligned_ocsvm_preds = []
        aligned_iforest_preds = []
        aligned_lstm_preds = []
        aligned_true_labels = []
        
        for discharge_id in all_ids:
            ocsvm_idx = ids_ocsvm_test.index(discharge_id)
            iforest_idx = ids_iforest_test.index(discharge_id)
            lstm_idx = ids_lstm_test.index(discharge_id)
            
            aligned_ocsvm_preds.append(ocsvm_predictions[ocsvm_idx])
            aligned_iforest_preds.append(iforest_predictions[iforest_idx])
            aligned_lstm_preds.append(lstm_predictions[lstm_idx])
            aligned_true_labels.append(y_ocsvm_test[ocsvm_idx])  # All y_*_test should have the same label for each ID
        
        # Convert to numpy arrays
        aligned_ocsvm_preds = np.array(aligned_ocsvm_preds)
        aligned_iforest_preds = np.array(aligned_iforest_preds)
        aligned_lstm_preds = np.array(aligned_lstm_preds)
        aligned_true_labels = np.array(aligned_true_labels)
        
        # Combine predictions
        combined_predictions = self.combine_predictions(
            aligned_ocsvm_preds, aligned_iforest_preds, aligned_lstm_preds
        )
        
        print("\nCombined Model Results:")
        print(classification_report(aligned_true_labels, combined_predictions))
        
        # Save test set details for later analysis
        self.test_data = {
            'X_ocsvm_test': X_ocsvm_test,
            'X_iforest_test': X_iforest_test,
            'X_lstm_test': X_lstm_test,
            'y_test': y_ocsvm_test,  # Use any of the y_*_test
            'ids_test': all_ids
        }

        return {
            'ocsvm': aligned_ocsvm_preds,
            'iforest': aligned_iforest_preds,
            'lstm': aligned_lstm_preds,
            'combined': combined_predictions,
            'true': aligned_true_labels,
            'ids': all_ids
        }

    def predict_ocsvm(self, X):
        """Predict using OCSVM model."""
        return predict_with_ocsvm(self.ocsvm_model, X)
    
    def predict_iforest(self, X):
        """Predict using Isolation Forest."""
        return predict_with_iforest(self.iforest_model, X)
    
    def predict_lstm(self, X):
        """Predict using LSTM model."""
        return predict_with_lstm(self.lstm_model, X)
    
    def combine_predictions(self, ocsvm_pred, iforest_pred, lstm_pred):
        all_preds = np.vstack([ocsvm_pred, iforest_pred, lstm_pred])
        votes = np.sum(all_preds, axis=0)
        return (votes >= 2).astype(int)

    def predict_discharge(self, discharge_id):
        if discharge_id not in self.discharge_data:
            print(f"Discharge {discharge_id} not found in data")
            return None
            
        # Get features for each model
        X_ocsvm = prepare_ocsvm_features_for_single_discharge(
            self.discharge_data, self.advanced_features, discharge_id
        )
        
        X_iforest = prepare_iforest_features_for_single_discharge(
            self.discharge_data, discharge_id, self.iforest_scaler
        )
        
        X_lstm = prepare_lstm_features_for_single_discharge(
            self.discharge_data, discharge_id, self.lstm_scaler
        )
        
        # Get predictions from each model
        ocsvm_pred = self.predict_ocsvm(X_ocsvm)[0]
        iforest_pred = self.predict_iforest(X_iforest)[0]
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
        
        ids_test = self.test_data['ids_test']
        results = {}
        
        print("\nDetailed test set results:")
        for discharge_id in ids_test:
            result = self.predict_discharge(discharge_id)
            results[discharge_id] = result
            
        # Calculate overall metrics
        actuals = [results[id]['actual'] for id in ids_test]
        ocsvm_preds = [results[id]['ocsvm'] for id in ids_test]
        iforest_preds = [results[id]['iforest'] for id in ids_test]
        lstm_preds = [results[id]['lstm'] for id in ids_test]
        final_preds = [results[id]['final'] for id in ids_test]
        
        print("\nOverall Accuracy:")
        print(f"OCSVM: {accuracy_score(actuals, ocsvm_preds):.4f}")
        print(f"Isolation Forest: {accuracy_score(actuals, iforest_preds):.4f}")
        print(f"LSTM: {accuracy_score(actuals, lstm_preds):.4f}")
        print(f"Combined: {accuracy_score(actuals, final_preds):.4f}")


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
            precision = np.sum((predictions == 1) & (results['true'] == 1)) / np.sum(predictions == 1)
            recall = np.sum((predictions == 1) & (results['true'] == 1)) / np.sum(results['true'] == 1)
            print(f"{model_name}: Precision = {precision:.4f}, Recall = {recall:.4f}")
