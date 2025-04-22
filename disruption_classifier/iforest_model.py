import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def prepare_features_for_iforest(discharge_data, discharge_labels):
    """
    Extract features from time series data for Isolation Forest.
    
    Args:
        discharge_data: Dictionary of discharge data
        discharge_labels: Dictionary of discharge labels
        
    Returns:
        X_scaled: Scaled feature matrix
        y: Target labels
        discharge_ids: List of discharge IDs
        scaler: Fitted StandardScaler for future use
    """
    X = []
    y = []
    discharge_ids = []
    
    for discharge_id, features in discharge_data.items():
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
        y.append(discharge_labels[discharge_id])
        discharge_ids.append(discharge_id)
        
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, discharge_ids, scaler

def train_iforest_model(X_train, y_train, params=None, random_state=42):
    """
    Train the Isolation Forest model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        params: Dictionary of model parameters
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Isolation Forest model
    """
    # Filter to use only non-disruptive cases
    X_train_normal = X_train[y_train == 0]
    
    # Use default parameters if not provided
    if params is None:
        params = {'n_estimators': 100, 'contamination': 0.1, 'max_features': 1.0}
    
    # Print Isolation Forest parameters
    print("\n" + "="*50)
    print("Isolation Forest Parameters:")
    print(f"  - n_estimators: {params['n_estimators']}")
    print(f"  - contamination: {params['contamination']}")
    print(f"  - max_features: {params['max_features']}")
    print("="*50)
    
    # Create and train the model
    model = IsolationForest(
        random_state=random_state, 
        n_estimators=int(params['n_estimators']),
        contamination=float(params['contamination']), 
        max_features=float(params['max_features'])
    )
    model.fit(X_train_normal)
    
    return model

def predict_with_iforest(model, X):
    """
    Make predictions using the Isolation Forest model.
    
    Args:
        model: Trained Isolation Forest model
        X: Features to predict
        
    Returns:
        Binary predictions (0 for normal, 1 for anomaly/disruption)
    """
    # Isolation Forest returns: 1 for normal, -1 for anomaly
    # We want: 0 for normal, 1 for anomaly/disruption
    raw_pred = model.predict(X)
    return (raw_pred == -1).astype(int)

def prepare_iforest_features_for_single_discharge(discharge_data, discharge_id, scaler):
    """
    Prepare features for a single discharge for Isolation Forest.
    
    Args:
        discharge_data: Dictionary of discharge data
        discharge_id: ID of the discharge to prepare features for
        scaler: Fitted StandardScaler
        
    Returns:
        Scaled feature vector ready for IForest prediction
    """
    features = {}
    for feature_num, feature_data in discharge_data[discharge_id].items():
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
    X_scaled = scaler.transform(X)
    
    return X_scaled
