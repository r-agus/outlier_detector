import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

def prepare_features_for_lstm(discharge_data, discharge_labels):
    """
    Extract and prepare features for LSTM model.
    
    Args:
        discharge_data: Dictionary of discharge data
        discharge_labels: Dictionary of discharge labels
        
    Returns:
        X_lstm: Feature matrix reshaped for LSTM
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
    
    # Reshape for LSTM: [samples, time steps, features]
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    return X_lstm, y, discharge_ids, scaler

def reshape_for_lstm(X):
    """
    Reshape data for LSTM input.
    
    Args:
        X: Feature matrix
        
    Returns:
        X_lstm: Reshaped features for LSTM
    """
    # Reshape data for LSTM: [samples, time steps, features]
    X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X_lstm

def train_lstm_model(X_train, y_train, params=None):
    """
    Train the LSTM model.
    
    Args:
        X_train: Training feature matrix (reshaped for LSTM)
        y_train: Training labels
        params: Dictionary of model parameters
        
    Returns:
        Trained LSTM model
    """
    # Use default parameters if not provided
    if params is None:
        params = {
            'layer1_units': 64,
            'layer2_units': 32,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 100
        }
    
    # Print LSTM parameters
    print("\n" + "="*50)
    print("LSTM Network Parameters:")
    print(f"  - layer1_units: {params['layer1_units']}")
    print(f"  - layer2_units: {params['layer2_units']}")
    print(f"  - dropout_rate: {params['dropout_rate']}")
    print(f"  - learning_rate: {params['learning_rate']}")
    print(f"  - batch_size: {params['batch_size']}")
    print(f"  - epochs: {params['epochs']}")
    print("="*50 + "\n")
    
    # LSTM model for binary classification
    model = Sequential([
        LSTM(params['layer1_units'], input_shape=(1, X_train.shape[2]), return_sequences=True),
        Dropout(params['dropout_rate']),
        LSTM(params['layer2_units']),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Use Adam optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        X_train, y_train, 
        epochs=params['epochs'], 
        batch_size=params['batch_size'], 
        verbose=1, 
        validation_split=0.2
    )
    
    return model

def predict_with_lstm(model, X):
    """
    Make predictions using the LSTM model.
    
    Args:
        model: Trained LSTM model
        X: Features to predict (reshaped for LSTM)
        
    Returns:
        Binary predictions (0 for normal, 1 for anomaly/disruption)
    """
    raw_pred = model.predict(X)
    return (raw_pred > 0.5).astype(int).flatten()

def prepare_lstm_features_for_single_discharge(discharge_data, discharge_id, scaler):
    """
    Prepare LSTM features for a single discharge.
    
    Args:
        discharge_data: Dictionary of discharge data
        discharge_id: ID of the discharge to prepare features for
        scaler: Fitted StandardScaler
        
    Returns:
        Feature vector reshaped for LSTM prediction
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
    
    # Reshape for LSTM
    X_lstm = reshape_for_lstm(X_scaled)
    
    return X_lstm
