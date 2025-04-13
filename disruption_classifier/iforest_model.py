import numpy as np
from sklearn.ensemble import IsolationForest

def prepare_features_for_iforest(X, y, discharge_ids):
    """
    Use the same features as currently being used.
    Just a pass-through function for consistency.
    """
    return X, y, discharge_ids

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
