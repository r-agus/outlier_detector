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
