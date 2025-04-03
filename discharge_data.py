"""
Module to hadle discharge file data.

This module provides functions to process time-to-disruption labels and
managing training/testing splits.
"""

import os
import pandas as pd
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DISCHARGE_DATA_PATH = 'C:\\Users\\Ruben\\Documents\\python-disruption-predictor\\remuestreados' # Path to the directory containing discharge data files
DISCHARGES_ETIQUETATION_DATA_FILE = '.\\rust-disruptor-predictor\\discharges-c23.txt' # Path to the file containing discharge etiquetation data, which includes the discharge and the time when the disruption happened, or 0 if it didn't happen.


# Function to load discharge data from a file
def load_discharge_data(file_path: str) -> pd.DataFrame:
    """
    Load discharge data from a txt file.

    The file should be in the format:
      <seconds since the beginning of the discharge> <discharge value>
    Both values are represented in scientific notation.

    Args:
        file_path (str): Path to the discharge data file.
    
    Returns:
        pd.DataFrame: DataFrame containing the discharge data with columns 'seconds' and 'discharge'.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    discharge_data = pd.read_table(file_path, sep='\\s+', header=None, names=['seconds', 'discharge'])
    discharge_data = discharge_data.dropna()  # Drop rows with NaN values
    discharge_data['seconds'] = discharge_data['seconds'].astype(float)
    discharge_data['value'] = discharge_data['discharge'].astype(float)
    return discharge_data


# Function to load all discharge data files from a directory
def load_all_discharge_data(directory: str) -> pd.DataFrame:
    """
    Load all discharge data files from a directory.

    Args:
        directory (str): Path to the directory containing discharge data files.
    
    Returns:
        pd.DataFrame: DataFrame containing all discharge data with columns 'seconds' and 'discharge'.
    """
    all_discharge_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            discharge_data = load_discharge_data(file_path)
            discharge_data['file'] = filename
            all_discharge_data = pd.concat([all_discharge_data, discharge_data], ignore_index=True)
    return all_discharge_data


# Function to load discharge data files that match a specific pattern
def load_discharge_data_by_pattern(directory: str, pattern: str) -> pd.DataFrame:
    """
    Load discharge data files that match a specific pattern from a directory.

    Args:
        directory (str): Path to the directory containing discharge data files.
        pattern (str): Pattern to match filenames.
    
    Returns:
        pd.DataFrame: DataFrame containing all matching discharge data with columns 'seconds' and 'discharge'.
    """
    all_discharge_data = pd.DataFrame()
    for filename in glob.glob(os.path.join(directory, pattern)):
        discharge_data = load_discharge_data(filename)
        discharge_data['file'] = os.path.basename(filename)  # Add a column with the filename
        all_discharge_data = pd.concat([all_discharge_data, discharge_data], ignore_index=True)
    return all_discharge_data


# Function to load discharge etiquetation data
def load_discharge_etiquetation_data(file_path: str) -> pd.DataFrame:
    """
    Load discharge etiquetation data from a file.

    Args:
        file_path (str): Path to the discharge etiquetation data file.
    
    Returns:
        pd.DataFrame: DataFrame containing the discharge etiquetation data.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_table(file_path, sep='\\s+', header=None, names=['discharge', 'time'])
    data = data.dropna()  # Drop rows with NaN values
    data['discharge'] = data['discharge'].astype(float)
    data['time'] = data['time'].astype(float)
    return data


if __name__ == "__main__":
    # Example usage
    discharge_data = load_all_discharge_data(DISCHARGE_DATA_PATH)
    logger.info(f"Loaded {len(discharge_data)} rows of discharge data from {DISCHARGE_DATA_PATH}.")
    logger.info(f"First 5 rows:\n{discharge_data.head()}")

    # Load discharge data by pattern
    pattern = 'DES_*_01_*.txt'
    discharge_data_pattern = load_discharge_data_by_pattern(DISCHARGE_DATA_PATH, pattern)
    logger.info(f"Loaded {len(discharge_data_pattern)} rows of discharge data matching pattern '{pattern}'.")
    logger.info(f"First 5 rows:\n{discharge_data_pattern.head()}")

    # Load discharge etiquetation data
    etiquetation_data = load_discharge_etiquetation_data(DISCHARGES_ETIQUETATION_DATA_FILE)
    logger.info(f"Loaded {len(etiquetation_data)} rows of discharge etiquetation data from {DISCHARGES_ETIQUETATION_DATA_FILE}.")
    logger.info(f"First rows:\n{etiquetation_data.head(10)}")


    