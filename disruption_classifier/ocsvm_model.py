import numpy as np
from sklearn.svm import OneClassSVM

def prepare_features_for_ocsvm(extracted_features, discharge_labels):
    """
    Prepare features specifically for OCSVM using the extracted features.
    
    Args:
        extracted_features: Dictionary of extracted features by discharge_id
        discharge_labels: Dictionary of labels by discharge_id
        
    Returns:
        X_scaled: Feature matrix
        y: Target labels
        discharge_ids: List of discharge IDs
    """
    X = []
    y = []
    discharge_ids = []
    
    for discharge_id, features in extracted_features.items():
        # Create a flat feature vector for this discharge
        feature_vector = []
        
        # Collect features for each feature type (1-7)
        for feature_num in range(1, 8):
            if feature_num in features:
                feature_vector.extend(features[feature_num])
            else:
                # If feature is missing, add zeros
                feature_vector.extend([0, 0, 0, 0, 0, 0])  # 6 values per feature
        
        X.append(feature_vector)
        y.append(discharge_labels[discharge_id])
        discharge_ids.append(discharge_id)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y, discharge_ids

def train_ocsvm_model(X_train, y_train, params=None):
    """
    Train the One-Class SVM model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        params: Dictionary of model parameters
        
    Returns:
        Trained OCSVM model
    """
    # Filter to use only non-disruptive cases
    X_train_normal = X_train[y_train == 0]
    
    # Use default parameters if not provided
    if params is None:
        params = {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'auto'}
    
    # Print OCSVM parameters
    print("\n" + "="*50)
    print("OCSVM Parameters:")
    print(f"  - kernel: {params['kernel']}")
    print(f"  - nu: {params['nu']}")
    print(f"  - gamma: {params['gamma']}")
    print("="*50)
    
    # Create and train the model
    model = OneClassSVM(
        kernel=params['kernel'], 
        nu=params['nu'], 
        gamma=params['gamma']
    )
    model.fit(X_train_normal)
    
    return model

def predict_with_ocsvm(model, X):
    """
    Make predictions using the OCSVM model.
    
    Args:
        model: Trained OCSVM model
        X: Features to predict
        
    Returns:
        Binary predictions (0 for normal, 1 for anomaly/disruption)
    """
    # OCSVM returns: 1 for normal, -1 for anomaly
    # We want: 0 for normal, 1 for anomaly/disruption
    raw_pred = model.predict(X)
    return (raw_pred == -1).astype(int)  # Convert to binary labels
