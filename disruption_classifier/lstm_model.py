import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_features_for_lstm(X, y):
    """
    Prepare features for LSTM model.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        X_lstm: Reshaped features for LSTM
        y: Target labels
    """
    # Reshape data for LSTM: [samples, time steps, features]
    X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X_lstm, y

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
