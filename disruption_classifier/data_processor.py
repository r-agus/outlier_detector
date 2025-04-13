import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def group_by_feature(discharge_data):
    """
    Group discharge data by feature number.
    
    Args:
        discharge_data: Dictionary of discharge data, keyed by discharge_id
        
    Returns:
        Dictionary of feature data grouped by feature number
    """
    feature_groups = {}
    
    for discharge_id, features in discharge_data.items():
        for feature_num, feature_data in features.items():
            if feature_num not in feature_groups:
                feature_groups[feature_num] = []
                
            # Add the feature data along with its discharge ID
            feature_groups[feature_num].append({
                'discharge_id': discharge_id,
                'data': feature_data
            })
    
    return feature_groups

def normalize_feature_groups(feature_groups, debug_output=False):
    """
    Normalize data within each feature group.
    
    Args:
        feature_groups: Dictionary of feature data grouped by feature number
        debug_output: Whether to write debug files
        
    Returns:
        Dictionary of normalized feature data
    """
    normalized_groups = {}
    
    for feature_num, feature_group in feature_groups.items():
        # Collect all values for this feature across all discharges
        all_values = np.concatenate([item['data']['value'].values for item in feature_group])
        
        # Create a scaler for this feature group
        scaler = StandardScaler()
        scaler.fit(all_values.reshape(-1, 1))
        
        # Normalize each discharge's data for this feature
        normalized_group = []
        for item in feature_group:
            normalized_values = scaler.transform(item['data']['value'].values.reshape(-1, 1)).flatten()
            normalized_data = pd.DataFrame({
                'time': item['data']['time'],
                'value': normalized_values
            })
            
            normalized_group.append({
                'discharge_id': item['discharge_id'],
                'data': normalized_data
            })
        
        normalized_groups[feature_num] = normalized_group
        
        # Debug output if requested
        if debug_output:
            debug_dir = 'debug_normalized'
            os.makedirs(debug_dir, exist_ok=True)
            
            for item in normalized_group:
                filename = f"{debug_dir}/DES_{item['discharge_id']}_{feature_num:02d}_normalized.txt"
                item['data'].to_csv(filename, sep=' ', index=False, header=False)
    
    return normalized_groups

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
                
                # Use a few representative values from the FFT
                # Taking the first 5 frequency components
                fft_features = fft_magnitude[:5] if len(fft_magnitude) >= 5 else np.pad(fft_magnitude, (0, 5-len(fft_magnitude)))
                
                # Combine features
                window_features.append(np.concatenate([[window_mean], fft_features]))
            
            # Average the features across all windows for this discharge and feature
            if window_features:
                extracted_features[discharge_id][feature_num] = np.mean(window_features, axis=0)
            else:
                # Handle case with not enough data for a window
                extracted_features[discharge_id][feature_num] = np.zeros(6)  # mean + 5 FFT components
    
    return extracted_features
