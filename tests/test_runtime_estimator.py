import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.models.runtime_estimator import RuntimeEstimator

@pytest.fixture
def runtime_estimator(tmp_path):
    """Create a RuntimeEstimator instance with temporary model directory"""
    return RuntimeEstimator(model_dir=tmp_path)

@pytest.fixture
def sample_task_data():
    """Create sample task data for testing"""
    return pd.DataFrame({
        'job_id': range(100),
        'scheduling_class': np.random.randint(0, 3, 100),
        'priority': np.random.randint(0, 5, 100),
        'cpu_request': np.random.uniform(0.1, 4.0, 100),
        'memory_request': np.random.uniform(0.1, 8.0, 100),
        'runtime': np.random.uniform(10, 1000, 100)
    })

@pytest.fixture
def sample_task_states():
    """Create sample task states for testing"""
    base_time = pd.Timestamp('2024-01-01')
    
    return pd.DataFrame({
        'task_id': range(5),
        'state': ['RUNNING'] * 3 + ['PENDING'] * 2,
        'start_time': [base_time - pd.Timedelta(minutes=i) for i in range(5)],
        'estimated_runtime': [100.0, 200.0, 300.0, 400.0, 500.0],
        'completion_ratio': [0.5, 0.75, None, None, None]
    })

def test_model_training(runtime_estimator, sample_task_data):
    """Test model training process"""
    runtime_estimator.train(sample_task_data)
    
    # Check if model files were created
    model_path = runtime_estimator.model_dir / 'runtime_model.joblib'
    scaler_path = runtime_estimator.model_dir / 'feature_scaler.joblib'
    
    assert model_path.exists()
    assert scaler_path.exists()
    assert runtime_estimator.model is not None

def test_runtime_prediction(runtime_estimator, sample_task_data):
    """Test runtime prediction"""
    # Train the model
    runtime_estimator.train(sample_task_data)
    
    # Make predictions
    predictions = runtime_estimator.predict(sample_task_data)
    
    assert len(predictions) == len(sample_task_data)
    assert all(predictions > 0)  # All predictions should be positive

def test_runtime_adjustment():
    """Test runtime estimate adjustment"""
    estimator = RuntimeEstimator()
    
    # Test case 1: Elapsed time exceeds initial estimate
    adjusted = estimator.adjust_runtime_estimate(
        initial_estimate=100,
        elapsed_time=150
    )
    assert adjusted == 300  # Should double elapsed time
    
    # Test case 2: With completion ratio
    adjusted = estimator.adjust_runtime_estimate(
        initial_estimate=100,
        elapsed_time=40,
        completion_ratio=0.5
    )
    assert adjusted == 100  # Should estimate based on progress
    
    # Test case 3: Without completion ratio, not exceeded
    adjusted = estimator.adjust_runtime_estimate(
        initial_estimate=100,
        elapsed_time=40
    )
    assert adjusted >= 90  # Should leave reasonable remaining time

def test_runtime_bin_assignment(runtime_estimator):
    """Test assignment of tasks to runtime bins"""
    bin_edges = np.array([0, 10, 100, 1000])
    
    assert runtime_estimator.get_runtime_bin(5, bin_edges) == 0
    assert runtime_estimator.get_runtime_bin(50, bin_edges) == 1
    assert runtime_estimator.get_runtime_bin(500, bin_edges) == 2
    assert runtime_estimator.get_runtime_bin(1500, bin_edges) == 3

def test_estimation_accuracy_analysis(runtime_estimator):
    """Test analysis of estimation accuracy"""
    actual = np.array([100, 200, 300, 400, 500])
    predicted = np.array([110, 180, 320, 380, 510])
    
    metrics = runtime_estimator.analyze_estimation_accuracy(actual, predicted)
    
    assert set(metrics.keys()) == {
        'mean_absolute_error',
        'median_absolute_error',
        'mean_relative_error',
        'median_relative_error',
        'rmse',
        'overestimation_rate'
    }
    assert all(isinstance(v, float) for v in metrics.values())

def test_update_estimates(runtime_estimator, sample_task_states):
    """Test updating runtime estimates for running tasks"""
    current_time = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=10)
    
    updated_states = runtime_estimator.update_estimates(
        sample_task_states,
        current_time
    )
    
    # Check that only running tasks were updated
    running_tasks = updated_states[updated_states['state'] == 'RUNNING']
    pending_tasks = updated_states[updated_states['state'] == 'PENDING']
    
    assert len(running_tasks) == 3
    assert len(pending_tasks) == 2
    
    # Check that estimates were adjusted
    assert all(
        running_tasks['estimated_runtime'] !=
        sample_task_states[sample_task_states['state'] == 'RUNNING']['estimated_runtime']
    )
    
    # Check that pending tasks weren't modified
    assert all(
        pending_tasks['estimated_runtime'] ==
        sample_task_states[sample_task_states['state'] == 'PENDING']['estimated_runtime']
    )

def test_model_persistence(runtime_estimator, sample_task_data, tmp_path):
    """Test saving and loading the model"""
    # Train and save model
    runtime_estimator.train(sample_task_data)
    
    # Create new estimator instance
    new_estimator = RuntimeEstimator(model_dir=tmp_path)
    
    # Load model
    assert new_estimator.load_model()
    
    # Make predictions with both models
    pred1 = runtime_estimator.predict(sample_task_data)
    pred2 = new_estimator.predict(sample_task_data)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(pred1, pred2)

def test_invalid_model_loading(tmp_path):
    """Test handling of invalid model loading"""
    estimator = RuntimeEstimator(model_dir=tmp_path)
    assert not estimator.load_model()  # Should return False when no model exists
    
    with pytest.raises(RuntimeError):
        # Should raise error when trying to predict without a model
        estimator.predict(pd.DataFrame({
            'scheduling_class': [1],
            'priority': [1],
            'cpu_request': [1.0],
            'memory_request': [1.0]
        }))

def test_feature_preparation(runtime_estimator, sample_task_data):
    """Test feature preparation for model training"""
    X, feature_names = runtime_estimator.prepare_features(sample_task_data)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(feature_names, list)
    assert X.shape[1] == len(feature_names)
    assert all(col in feature_names for col in runtime_estimator.feature_columns)

def test_empty_data_handling(runtime_estimator):
    """Test handling of empty input data"""
    empty_df = pd.DataFrame(columns=runtime_estimator.feature_columns)
    
    X, feature_names = runtime_estimator.prepare_features(empty_df)
    assert X.shape[0] == 0
    assert len(feature_names) == len(runtime_estimator.feature_columns)

def test_missing_features_handling(runtime_estimator):
    """Test handling of missing feature columns"""
    incomplete_data = pd.DataFrame({
        'scheduling_class': [1, 2],
        'priority': [1, 2]
        # Missing cpu_request and memory_request
    })
    
    X, feature_names = runtime_estimator.prepare_features(incomplete_data)
    assert X.shape[1] == 2  # Should only include available features
    assert len(feature_names) == 2