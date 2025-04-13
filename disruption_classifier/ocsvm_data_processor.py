from locale import normalize
import os
from threading import local
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt

def load_and_group_by_feature(data_path, discharge_ids=None, features_to_use=range(1, 8)):
    """
    Load and group discharge data by feature number.
    
    Args:
        data_path: Path to directory containing discharge data files
        discharge_ids: List of discharge IDs to process (if None, process all)
        features_to_use: Range of feature numbers to use (default 1-7)
        
    Returns:
        Dictionary with feature numbers as keys and lists of data as values
    """
    # Dictionary to store data grouped by feature
    feature_groups = defaultdict(list)
    
    # Regular expression to extract discharge ID and feature number from filename
    pattern = r"DES_(\d+)_(\d+)_"
    
    for filename in os.listdir(data_path):
        match = re.search(pattern, filename)
        if match:
            discharge_id = match.group(1)
            feature_num = int(match.group(2))
            
            # Check if this is a discharge ID we want to process
            if (discharge_ids is None or discharge_id in discharge_ids) and feature_num in features_to_use:
                filepath = os.path.join(data_path, filename)
                try:
                    # Load the data
                    data = pd.read_csv(filepath, sep='\s+', header=None, names=['time', 'value'])
                    # Store the data with metadata
                    feature_groups[feature_num].append({
                        'discharge_id': discharge_id,
                        'data': data
                    })
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
    
    return feature_groups

def normalize_feature_groups(feature_groups, output_dir=None):
    """
    Normalize data within each feature group.
    
    Args:
        feature_groups: Dictionary with feature numbers as keys and lists of data as values
        output_dir: Directory to save normalized data for debugging (optional)
        
    Returns:
        Dictionary with normalized feature groups
    """
    normalized_groups = {}
    
    for feature_num, data_list in feature_groups.items():
        # Combine all values for this feature
        all_values = np.concatenate([item['data']['value'].values for item in data_list])
        
        local_min = np.min(all_values)
        local_max = np.max(all_values)

        normalized_data_list = []
        
        for item in data_list:

            normalized_item = []

            for i in range(len(item['data'])):
                normalized_value = (item['data']['value'].values[i] - local_min) / (local_max - local_min)
                normalized_item.append(normalized_value)
    
            normalized_data_list.append(normalized_item)
        
        normalized_groups[int(feature_num)] = normalized_data_list

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i in enumerate(normalized_data_list):
                normalized_df = pd.DataFrame(i[1], columns=['normalized_value'])
                normalized_df.to_csv(os.path.join(output_dir, f"normalized_feature_{feature_num}.csv"), index=False)
    
    return normalized_groups

def extract_window_features(normalized_groups, window_size=16):
    """
    Extract features from windows of normalized data:
    1. Mean value across windows
    2. FFT features aggregated across windows
    
    Args:
        normalized_groups: Dictionary with normalized feature groups
        window_size: Size of windows for feature extraction
        
    Returns:
        Dictionary mapping each discharge_id to its extracted features
    """
    discharge_features = {}
    
    for feature_num, data_list in normalized_groups.items():
        for data in data_list:            
            # Process data in windows
            num_windows = len(data) // window_size
            
            if num_windows == 0:
                continue  # Skip if not enough data for even one window
                
            # Collect statistics across all windows
            window_means = []
            window_fft_means = []
            window_fft_stds = []
            window_fft_maxes = []
            
            for i in range(num_windows):
                window = data[i * window_size:(i + 1) * window_size]
                
                # Mean value of window
                window_means.append(np.mean(window))
                
                # FFT of window (excluding DC component)
                fft_result = np.fft.fft(window)
                fft_magnitude = np.abs(fft_result[1:])  # Remove DC component
                
                window_fft_means.append(np.mean(fft_magnitude))
                window_fft_stds.append(np.std(fft_magnitude))
                window_fft_maxes.append(np.max(fft_magnitude))
            
            # Calculate aggregate statistics for this feature
            feature_stats = [
                np.mean(window_means),
                np.std(window_means),
                np.mean(window_fft_means),
                np.std(window_fft_means),
                np.mean(window_fft_stds),
                np.mean(window_fft_maxes)
            ]
            
            # Add aggregated feature statistics for this feature number
            discharge_features[discharge_id].extend(feature_stats)
    
    return discharge_features

def process_data_for_classifier(data_path, discharge_ids=None, features_to_use=range(1, 8), 
                               window_size=16, debug_output=None):
    """
    Complete pipeline to process discharge data for the classifier.
    
    Args:
        data_path: Path to directory containing discharge data files
        discharge_ids: List of discharge IDs to process
        features_to_use: Range of feature numbers to use
        window_size: Size of windows for feature extraction
        debug_output: Directory to save intermediate results for debugging (optional)
        
    Returns:
        Dictionary mapping each discharge_id to its extracted features
    """
    # Step 1: Group data by feature
    feature_groups = load_and_group_by_feature(data_path, discharge_ids, features_to_use)
    
    # Step 2: Normalize data within each feature group
    normalized_groups = normalize_feature_groups(feature_groups, debug_output)
    
    # Step 3: Extract features from windows of normalized data
    discharge_features = extract_window_features(normalized_groups, window_size)
    
    return discharge_features
