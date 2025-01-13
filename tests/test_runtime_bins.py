import pytest
import pandas as pd
import numpy as np
from src.models.runtime_bins import RuntimeBinsManager

@pytest.fixture
def bins_manager():
    """Create a RuntimeBinsManager instance"""
    return RuntimeBinsManager(n_bins=5)

@pytest.fixture
def sample_runtimes():
    """Create sample runtime data"""
    # Create log-normal distribution of runtimes
    return pd.Series(np.random.lognormal(mean=4.0, sigma=1.0, size=1000))

def test_bin_creation(bins_manager, sample_runtimes):
    """Test creation of runtime bins"""
    bins_manager.create_bins(sample_runtimes)
    
    # Check number of bins
    assert len(bins_manager.bins) == 5
    
    # Check bin properties
    for i, bin_ in enumerate(bins_manager.bins):
        assert bin_.index == i
        assert bin_.start < bin_.end
        assert isinstance(bin_.label, str)
        assert bin_.task_count == 0
        assert bin_.total_cpu == 0.0
        assert bin_.total_memory == 0.0

def test_bin_index_assignment(bins_manager, sample_runtimes):
    """Test assignment of runtimes to bin indices"""
    bins_manager.create_bins(sample_runtimes)
    
    # Test various runtime values
    min_runtime = sample_runtimes.min()
    max_runtime = sample_runtimes.max()
    mid_runtime = np.exp((np.log(min_runtime) + np.log(max_runtime)) / 2)
    
    # Check bin assignments
    assert bins_manager.get_bin_index(min_runtime) == 0
    assert bins_manager.get_bin_index(max_runtime) == 4  # last bin
    assert 0 <= bins_manager.get_bin_index(mid_runtime) <= 4

def test_task_addition(bins_manager, sample_runtimes):
    """Test adding tasks to bins"""
    bins_manager.create_bins(sample_runtimes)
    
    # Add some tasks
    tasks = [
        (100.0, 2.0, 4.0),  # runtime, cpu, memory
        (200.0, 1.0, 2.0),
        (300.0, 3.0, 6.0)
    ]
    
    for runtime, cpu, memory in tasks:
        bins_manager.add_task_to_bin(runtime, cpu, memory)
    
    # Get bin statistics
    stats = bins_manager.get_bin_stats()
    
    # Check that tasks were counted
    assert stats['task_count'].sum() == len(tasks)
    assert stats['total_cpu'].sum() == sum(task[1] for task in tasks)
    assert stats['total_memory'].sum() == sum(task[2] for task in tasks)

def test_bin_stats(bins_manager, sample_runtimes):
    """Test bin statistics calculation"""
    bins_manager.create_bins(sample_runtimes)
    
    # Add some tasks
    for _ in range(10):
        runtime = np.random.choice(sample_runtimes)
        cpu = np.random.uniform(1.0, 4.0)
        memory = np.random.uniform(2.0, 8.0)
        bins_manager.add_task_to_bin(runtime, cpu, memory)
    
    stats = bins_manager.get_bin_stats()
    
    # Check stats structure
    expected_columns = {
        'bin_index', 'start_time', 'end_time', 'label',
        'task_count', 'total_cpu', 'total_memory',
        'avg_cpu_per_task', 'avg_memory_per_task'
    }
    assert set(stats.columns) == expected_columns
    
    # Check stats validity
    assert all(stats['task_count'] >= 0)
    assert all(stats['total_cpu'] >= 0)
    assert all(stats['total_memory'] >= 0)
    assert all(stats['avg_cpu_per_task'] >= 0)
    assert all(stats['avg_memory_per_task'] >= 0)

def test_distribution_analysis(bins_manager, sample_runtimes):
    """Test analysis of bin distribution"""
    bins_manager.create_bins(sample_runtimes)
    
    # Add tasks following a normal distribution
    n_tasks = 100
    mean_bin = 2  # Middle bin
    bin_indices = np.random.normal(mean_bin, 1.0, n_tasks)
    bin_indices = np.clip(bin_indices, 0, 4).astype(int)
    
    for bin_idx in bin_indices:
        runtime = bins_manager.bins[bin_idx].start + 1  # Just after start of bin
        bins_manager.add_task_to_bin(runtime, 1.0, 2.0)
    
    analysis = bins_manager.analyze_bin_distribution()
    
    # Check analysis structure
    expected_metrics = {
        'entropy', 'most_common_bin', 'empty_bins', 'max_bin_utilization'
    }
    assert set(analysis.keys()) == expected_metrics
    
    # Check metric validity
    assert 0 <= analysis['entropy'] <= np.log2(5)  # Max entropy for 5 bins
    assert 0 <= analysis['most_common_bin'] <= 4
    assert 0 <= analysis['empty_bins'] <= 5
    assert 0 <= analysis['max_bin_utilization'] <= 1.0

def test_empty_data_handling(bins_manager):
    """Test handling of empty input data"""
    # Empty series
    empty_series = pd.Series([])
    bins_manager.create_bins(empty_series)
    assert len(bins_manager.bins) == 0
    
    # No tasks added
    bins_manager.create_bins(pd.Series([1.0, 2.0, 3.0]))
    stats = bins_manager.get_bin_stats()
    assert all(stats['task_count'] == 0)
    assert all(stats['total_cpu'] == 0)
    assert all(stats['total_memory'] == 0)

def test_bin_clearing(bins_manager, sample_runtimes):
    """Test clearing of bin statistics"""
    bins_manager.create_bins(sample_runtimes)
    
    # Add some tasks
    for _ in range(5):
        runtime = np.random.choice(sample_runtimes)
        bins_manager.add_task_to_bin(runtime, 1.0, 2.0)
    
    # Verify tasks were added
    stats_before = bins_manager.get_bin_stats()
    assert stats_before['task_count'].sum() > 0
    
    # Clear bins
    bins_manager.clear_bins()
    
    # Check that all counts were reset
    stats_after = bins_manager.get_bin_stats()
    assert all(stats_after['task_count'] == 0)
    assert all(stats_after['total_cpu'] == 0)
    assert all(stats_after['total_memory'] == 0)

def test_bin_edges_and_labels(bins_manager, sample_runtimes):
    """Test retrieval of bin edges and labels"""
    bins_manager.create_bins(sample_runtimes)
    
    # Check bin edges
    edges = bins_manager.get_bin_edges()
    assert len(edges) == 6  # n_bins + 1
    assert all(edges[i] < edges[i+1] for i in range(len(edges)-1))
    
    # Check bin labels
    labels = bins_manager.get_bin_labels()
    assert len(labels) == 5  # n_bins
    assert all(isinstance(label, str) for label in labels)
    assert all('Bin' in label for label in labels)