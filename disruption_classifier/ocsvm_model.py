import numpy as np
from sklearn.svm import OneClassSVM
from data_processor import group_by_feature, normalize_feature_groups

def extract_features_from_time_series(normalized_groups, window_size=16):
    """
    Extract advanced features from normalized time series data:
    1. Mean of window
    2. FFT of window without DC component
    
    Args:
        normalized_groups: Dictionary of normalized feature data
        window_size: Size of the window for feature extraction
        
    Returns:
        Dictionary mapping discharge_ids to extracted features
    """
    extracted_features = {}
    
    for feature_num, feature_group in normalized_groups.items():
        for item in feature_group:
            discharge_id = item['discharge_id']
            data = item['data']['value'].values
            
            if discharge_id not in extracted_features:
                extracted_features[discharge_id] = {}
            
            # Process data in windows
            num_windows = len(data) // window_size
            window_features = []
            
            for i in range(num_windows):
                window = data[i * window_size:(i + 1) * window_size]
                
                # Feature 1: Mean of window
                window_mean = np.mean(window)
                
                # Feature 2: FFT without DC component
                fft_result = np.fft.fft(window)
                fft_without_dc = fft_result[1:]  # Remove DC component
                fft_magnitude = np.abs(fft_without_dc)
                fft_energy = np.mean(fft_magnitude) if len(fft_magnitude) > 0 else 0  # TODO: Is this correct?
                # Combine features
                window_features.append(np.array([window_mean, fft_energy]))
            
            # Average the features across all windows for this discharge and feature
            if window_features:
                extracted_features[discharge_id][feature_num] = np.mean(window_features, axis=0)
            else:
                # Handle case with not enough data for a window
                extracted_features[discharge_id][feature_num] = np.zeros(2)  # mean + FFT energy
    
    return extracted_features

def prepare_features_for_ocsvm(discharge_data, discharge_labels, debug_output=False):
    """
    Complete pipeline for preparing OCSVM features from raw data.
    
    Args:
        discharge_data: Dictionary of discharge data
        discharge_labels: Dictionary of discharge labels
        debug_output: Whether to write debug files
        
    Returns:
        X: Feature matrix
        y: Target labels
        discharge_ids: List of discharge IDs
    """
    # Group by feature
    feature_groups = group_by_feature(discharge_data)
    
    # Normalize
    normalized_groups = normalize_feature_groups(feature_groups, debug_output)
    
    # Extract advanced features
    advanced_features = extract_features_from_time_series(normalized_groups)
    
    # Create feature matrix
    X = []
    y = []
    discharge_ids = []
    
    for discharge_id, features in advanced_features.items():
        # Create a flat feature vector for this discharge
        feature_vector = []
    
        # Collect features for each feature type (1-7)
        for feature_num in range(1, 8):
            if feature_num in features:
                feature_vector.extend(features[feature_num])
            else:
                # If feature is missing, add zeros
                feature_vector.extend([0, 0])  # 2 values per feature
        
        X.append(feature_vector)
        y.append(discharge_labels[discharge_id])
        discharge_ids.append(discharge_id)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y, discharge_ids, advanced_features

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
    
    # Use updated default parameters if not provided
    if params is None:
        params = {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'scale'}
    
    # Print OCSVM parameters
    print("\n" + "="*50)
    print("OCSVM Parameters:")
    print(f"  - kernel: {params['kernel']}")
    print(f"  - nu: {params['nu']}")
    print(f"  - gamma: {params['gamma']}")
    print(f"Training OCSVM on {X_train_normal.shape[0]} normal samples.")
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

def prepare_ocsvm_features_for_single_discharge(discharge_data, advanced_features, discharge_id):
    """
    Prepare OCSVM features for a single discharge.
    
    Args:
        discharge_data: Dictionary with all discharge data
        advanced_features: Pre-computed advanced features
        discharge_id: ID of the discharge to prepare features for
        
    Returns:
        Feature vector ready for OCSVM prediction
    """
    if advanced_features and discharge_id in advanced_features:
        ocsvm_features = []
        for i in range(1, 8):
            if i in advanced_features[discharge_id]:
                ocsvm_features.extend(advanced_features[discharge_id][i])
            else:
                ocsvm_features.extend([0, 0])
                
        X_ocsvm = np.array([ocsvm_features])
        return X_ocsvm
    else:
        # If advanced features aren't available, compute them on the fly
        # (This would be a simplified version for a single discharge)
        print(f"Warning: Advanced features not pre-computed for discharge {discharge_id}")
        feature_groups = {
            feature_num: [{'discharge_id': discharge_id, 'data': data}] 
            for feature_num, data in discharge_data[discharge_id].items()
        }
        
        normalized_groups = normalize_feature_groups(feature_groups)
        advanced_features = extract_features_from_time_series(normalized_groups)
        
        # Create feature vector
        feature_vector = []
        for feature_num in range(1, 8):
            if feature_num in advanced_features[discharge_id]:
                feature_vector.extend(advanced_features[discharge_id][feature_num])
            else:
                feature_vector.extend([0, 0])
                
        return np.array([feature_vector])
