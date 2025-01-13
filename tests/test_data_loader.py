import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader

@pytest.fixture
def data_loader():
    """Create a DataLoader instance pointing to the test data directory"""
    return DataLoader(Path("data"))

def test_machine_events_loading(data_loader):
    """Test loading machine events data"""
    df = data_loader.load_machine_events()
    
    # Check if DataFrame is not empty
    assert not df.empty
    
    # Check if all expected columns are present
    expected_cols = [
        'timestamp', 'machine_id', 'event_type', 
        'platform_id', 'cpu_capacity', 'memory_capacity'
    ]
    assert all(col in df.columns for col in expected_cols)
    
    # Check if event types are properly mapped
    assert all(df['event_type'].isin(['ADD', 'REMOVE', 'UPDATE']))
    
    # Check if timestamp is converted to datetime
    assert isinstance(df['timestamp'].iloc[0], pd.Timestamp)

def test_task_events_loading(data_loader):
    """Test loading task events data"""
    df = data_loader.load_task_events()
    
    # Check if DataFrame is not empty
    assert not df.empty
    
    # Check if all expected columns are present
    expected_cols = [
        'timestamp', 'missing_info', 'job_id', 'task_index',
        'machine_id', 'event_type', 'user_name', 'scheduling_class',
        'priority', 'cpu_request', 'memory_request', 'disk_request',
        'different_machine_constraint'
    ]
    assert all(col in df.columns for col in expected_cols)
    
    # Check if event types are properly mapped
    valid_events = [
        'SUBMIT', 'SCHEDULE', 'EVICT', 'FAIL', 'FINISH',
        'KILL', 'LOST', 'UPDATE_PENDING', 'UPDATE_RUNNING'
    ]
    assert all(df['event_type'].isin(valid_events))
    
    # Check if timestamp is converted to datetime
    assert isinstance(df['timestamp'].iloc[0], pd.Timestamp)

def test_sample_data_loading(data_loader):
    """Test loading sample data"""
    sample_size = 100
    data = data_loader.load_sample_data(n_tasks=sample_size)
    
    # Check if both machine and task events are loaded
    assert 'machine_events' in data
    assert 'task_events' in data
    
    # Check if machine events DataFrame is not empty
    assert not data['machine_events'].empty
    
    # Check if task events DataFrame is not empty
    assert not data['task_events'].empty
    
    # Check if number of unique jobs is less than or equal to requested sample size
    n_unique_jobs = data['task_events']['job_id'].nunique()
    assert n_unique_jobs <= sample_size

def test_invalid_directory():
    """Test handling of invalid data directory"""
    with pytest.raises(FileNotFoundError):
        DataLoader(Path("nonexistent_directory"))

def test_invalid_file_reading(data_loader, tmp_path):
    """Test handling of invalid file reading"""
    # Create an empty file
    invalid_file = tmp_path / "invalid.csv.gz"
    invalid_file.touch()
    
    with pytest.raises(Exception):
        data_loader.read_gzip_csv(
            invalid_file, 
            data_loader.MACHINE_COLS
        )