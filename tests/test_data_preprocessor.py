import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_preprocessor import DataPreprocessor

@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance"""
    return DataPreprocessor()

@pytest.fixture
def sample_task_events():
    """Create sample task events data"""
    base_time = pd.Timestamp('2024-01-01')
    
    return pd.DataFrame({
        'timestamp': [
            base_time,
            base_time + pd.Timedelta(seconds=10),
            base_time + pd.Timedelta(seconds=20),
            base_time + pd.Timedelta(seconds=30)
        ],
        'job_id': [1, 1, 2, 2],
        'task_index': [0, 0, 0, 0],
        'event_type': ['SUBMIT', 'FINISH', 'SUBMIT', 'FAIL'],
        'cpu_request': [1.0, 1.0, 2.0, 2.0],
        'memory_request': [2.0, 2.0, 4.0, 4.0],
        'scheduling_class': [1, 1, 2, 2],
        'priority': [0, 0, 1, 1]
    })

@pytest.fixture
def sample_machine_events():
    """Create sample machine events data"""
    base_time = pd.Timestamp('2024-01-01')
    
    return pd.DataFrame({
        'timestamp': [
            base_time,
            base_time + pd.Timedelta(seconds=10),
            base_time + pd.Timedelta(seconds=20)
        ],
        'machine_id': [1, 1, 1],
        'event_type': ['ADD', 'UPDATE', 'REMOVE'],
        'platform_id': [1, 1, 1],
        'cpu_capacity': [4.0, 8.0, 8.0],
        'memory_capacity': [8.0, 16.0, 16.0]
    })

def test_calculate_task_runtimes(preprocessor, sample_task_events):
    """Test calculation of task runtimes"""
    runtimes = preprocessor.calculate_task_runtimes(sample_task_events)
    
    assert len(runtimes) == 2  # Two tasks
    assert all(runtime > 0 for runtime in runtimes['runtime'])
    assert set(runtimes.columns) == {
        'job_id', 'task_index', 'start_time', 'end_time', 'runtime',
        'end_event_type', 'cpu_request', 'memory_request',
        'scheduling_class', 'priority'
    }

def test_process_machine_states(preprocessor, sample_machine_events):
    """Test processing of machine states"""
    states = preprocessor.process_machine_states(sample_machine_events)
    
    assert len(states) == 2  # Two state changes
    assert all(col in states.columns for col in [
        'machine_id', 'start_time', 'end_time', 'platform_id',
        'cpu_capacity', 'memory_capacity', 'state'
    ])
    
    # Check state transitions
    first_state = states.iloc[0]
    assert first_state['state'] == 'ACTIVE'
    assert first_state['cpu_capacity'] == 4.0
    
    last_state = states.iloc[-1]
    assert last_state['state'] == 'REMOVED'
    assert last_state['cpu_capacity'] == 8.0

def test_create_runtime_bins(preprocessor):
    """Test creation of runtime bins"""
    runtimes = pd.Series([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    bins = preprocessor.create_runtime_bins(runtimes, n_bins=5)
    
    assert len(bins) == 5  # Requested number of bins
    assert all(col in bins.columns for col in ['bin_start', 'bin_end', 'task_count', 'bin_label'])
    assert all(bins['bin_start'] < bins['bin_end'])  # Bins are properly ordered
    assert all(bins['task_count'] >= 0)  # No negative counts

def test_validate_data(preprocessor):
    """Test data validation"""
    # Create valid test data
    valid_task_runtimes = pd.DataFrame({
        'job_id': [1, 2],
        'task_index': [0, 0],
        'runtime': [10.0, 20.0],
        'cpu_request': [1.0, 2.0],
        'memory_request': [2.0, 4.0]
    })
    
    valid_machine_states = pd.DataFrame({
        'machine_id': [1, 1],
        'start_time': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-01 00:00:10')
        ],
        'end_time': [
            pd.Timestamp('2024-01-01 00:00:10'),
            pd.Timestamp('2024-01-01 00:00:20')
        ],
        'state': ['ACTIVE', 'REMOVED']
    })
    
    is_valid, messages = preprocessor.validate_data(
        valid_task_runtimes,
        valid_machine_states
    )
    assert is_valid
    assert len(messages) == 0

def test_validate_data_with_issues(preprocessor):
    """Test data validation with problematic data"""
    # Create invalid test data
    invalid_task_runtimes = pd.DataFrame({
        'job_id': [1],
        'task_index': [0],
        'runtime': [-10.0],  # Negative runtime
        'cpu_request': [None],  # Missing resource request
        'memory_request': [2.0]
    })
    
    invalid_machine_states = pd.DataFrame({
        'machine_id': [1, 1],
        'start_time': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-01 00:00:05')  # Overlapping state
        ],
        'end_time': [
            pd.Timestamp('2024-01-01 00:00:10'),
            pd.Timestamp('2024-01-01 00:00:15')
        ],
        'state': ['ACTIVE', 'REMOVED']
    })
    
    is_valid, messages = preprocessor.validate_data(
        invalid_task_runtimes,
        invalid_machine_states
    )
    assert not is_valid
    assert len(messages) > 0
    assert any('non-positive runtimes' in msg for msg in messages)
    assert any('missing resource requests' in msg for msg in messages)
    assert any('overlapping states' in msg for msg in messages)

def test_preprocess_data(preprocessor, sample_task_events, sample_machine_events):
    """Test complete data preprocessing pipeline"""
    result = preprocessor.preprocess_data(
        sample_task_events,
        sample_machine_events,
        n_runtime_bins=5
    )
    
    assert set(result.keys()) == {
        'task_runtimes',
        'machine_states',
        'runtime_bins'
    }
    
    # Check task runtimes
    assert not result['task_runtimes'].empty
    assert all(result['task_runtimes']['runtime'] > 0)
    
    # Check machine states
    assert not result['machine_states'].empty
    assert all(col in result['machine_states'].columns for col in [
        'machine_id', 'start_time', 'end_time', 'state'
    ])
    
    # Check runtime bins
    assert len(result['runtime_bins']) == 5
    assert all(col in result['runtime_bins'].columns for col in [
        'bin_start', 'bin_end', 'task_count', 'bin_label'
    ])

def test_empty_data_handling(preprocessor):
    """Test handling of empty input data"""
    empty_task_events = pd.DataFrame(columns=[
        'timestamp', 'job_id', 'task_index', 'event_type',
        'cpu_request', 'memory_request', 'scheduling_class', 'priority'
    ])
    
    empty_machine_events = pd.DataFrame(columns=[
        'timestamp', 'machine_id', 'event_type', 'platform_id',
        'cpu_capacity', 'memory_capacity'
    ])
    
    result = preprocessor.preprocess_data(
        empty_task_events,
        empty_machine_events
    )
    
    # Check that all expected DataFrames are returned
    assert isinstance(result, dict)
    assert set(result.keys()) == {'task_runtimes', 'machine_states', 'runtime_bins'}
    
    # Check task_runtimes
    assert result['task_runtimes'].empty
    assert set(result['task_runtimes'].columns) == {
        'job_id', 'task_index', 'start_time', 'end_time', 'runtime',
        'end_event_type', 'cpu_request', 'memory_request',
        'scheduling_class', 'priority'
    }
    
    # Check machine_states
    assert result['machine_states'].empty
    assert set(result['machine_states'].columns) == {
        'machine_id', 'start_time', 'end_time', 'platform_id',
        'cpu_capacity', 'memory_capacity', 'state'
    }
    
    # Check runtime_bins
    assert result['runtime_bins'].empty
    assert set(result['runtime_bins'].columns) == {
        'bin_start', 'bin_end', 'task_count', 'bin_label'
    }