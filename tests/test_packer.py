import pytest
import numpy as np
from src.models.packer import Packer, Task, Instance

@pytest.fixture
def packer():
    """Create a Packer instance"""
    packer = Packer(n_runtime_bins=5)
    # Set runtime bins [0, 10, 100, 1000, 10000]
    packer.set_runtime_bins([0, 10, 100, 1000, 10000])
    return packer

@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing"""
    return [
        Task("task1", "job1", 50, 2.0, 4.0, 1, 1),   # Medium runtime
        Task("task2", "job1", 500, 1.0, 2.0, 2, 1),  # Long runtime
        Task("task3", "job2", 5, 1.0, 2.0, 0, 2),    # Short runtime
        Task("task4", "job2", 200, 2.0, 4.0, 2, 2)   # Long runtime
    ]

@pytest.fixture
def sample_instances():
    """Create sample instances for testing"""
    return [
        Instance("i-1", 4.0, 8.0),  # Standard instance
        Instance("i-2", 8.0, 16.0)  # Large instance
    ]

def test_runtime_bin_assignment(packer):
    """Test assignment of runtimes to bins"""
    assert packer.get_runtime_bin(5) == 0    # First bin
    assert packer.get_runtime_bin(50) == 1   # Second bin
    assert packer.get_runtime_bin(500) == 2  # Third bin
    assert packer.get_runtime_bin(5000) == 3 # Fourth bin

def test_task_fitting(packer, sample_tasks, sample_instances):
    """Test checking if tasks can fit on instances"""
    task = sample_tasks[0]  # 2 CPU, 4 Memory
    
    # Should fit on both instances
    assert packer.can_fit_task(task, sample_instances[0])  # 4 CPU, 8 Memory
    assert packer.can_fit_task(task, sample_instances[1])  # 8 CPU, 16 Memory
    
    # Fill up first instance
    instance = sample_instances[0]
    instance.cpu_used = 3.0
    instance.memory_used = 6.0
    
    # Now task shouldn't fit
    assert not packer.can_fit_task(task, instance)

def test_instance_selection(packer, sample_tasks, sample_instances):
    """Test selection of best instance for tasks"""
    # Setup instances with different runtime bins
    instance1, instance2 = sample_instances
    instance1.runtime_bin = 1  # Medium runtime bin
    instance2.runtime_bin = 2  # Long runtime bin
    
    # Test same bin matching
    task = sample_tasks[0]  # Medium runtime (bin 1)
    best_instance = packer.find_best_instance(task, sample_instances, allow_down_packing=True)
    assert best_instance.instance_id == instance1.instance_id  # Should match same bin
    
    # Test up-packing
    task = sample_tasks[2]  # Short runtime (bin 0)
    best_instance = packer.find_best_instance(task, sample_instances, allow_down_packing=False)
    assert best_instance.instance_id == instance1.instance_id  # Should up-pack to bin 1
    
    # Test down-packing
    task = sample_tasks[1]  # Long runtime (bin 2)
    # First try without down-packing
    best_instance = packer.find_best_instance(task, [instance1], allow_down_packing=False)
    assert best_instance is None  # Should not find instance
    
    # Then try with down-packing
    best_instance = packer.find_best_instance(task, [instance1], allow_down_packing=True)
    assert best_instance.instance_id == instance1.instance_id  # Should down-pack to bin 1

def test_task_packing(packer, sample_tasks, sample_instances):
    """Test full task packing algorithm"""
    assignments = packer.pack_tasks(sample_tasks, sample_instances)
    
    # Check that all tasks were assigned
    assigned_tasks = sum(len(tasks) for tasks in assignments.values())
    assert assigned_tasks == len(sample_tasks)
    
    # Check instance constraints
    is_valid, messages = packer.validate_packing()
    assert is_valid, f"Packing validation failed: {messages}"
    
    # Check instance statistics
    stats = packer.get_instance_stats()
    assert len(stats) == len(sample_instances)
    assert all(stats['cpu_utilization'] <= 1.0)
    assert all(stats['memory_utilization'] <= 1.0)

def test_oversubscribed_packing(packer):
    """Test packing with insufficient resources"""
    # Create tasks that require more resources than available
    oversized_tasks = [
        Task("task1", "job1", 50, 4.0, 8.0, 1, 1),   # Uses entire small instance
        Task("task2", "job1", 50, 4.0, 8.0, 1, 1),   # Won't fit on small instance
        Task("task3", "job1", 50, 4.0, 8.0, 1, 1)    # Won't fit on small instance
    ]
    
    small_instance = [Instance("i-1", 4.0, 8.0)]
    
    assignments = packer.pack_tasks(oversized_tasks, small_instance)
    
    # Should only assign the first task
    assert sum(len(tasks) for tasks in assignments.values()) == 1
    
    # Validate packing
    is_valid, messages = packer.validate_packing()
    assert is_valid, f"Packing validation failed: {messages}"

def test_empty_inputs(packer, sample_instances, sample_tasks):
    """Test handling of empty inputs"""
    # Empty task list
    assignments = packer.pack_tasks([], sample_instances)
    assert len(assignments) == 0
    
    # Empty instance list
    assignments = packer.pack_tasks(sample_tasks, [])
    assert len(assignments) == 0
    
    # Both empty
    assignments = packer.pack_tasks([], [])
    assert len(assignments) == 0

def test_instance_stats(packer, sample_tasks, sample_instances):
    """Test instance statistics calculation"""
    # Pack some tasks
    packer.pack_tasks(sample_tasks, sample_instances)
    
    # Get statistics
    stats = packer.get_instance_stats()
    
    # Check stats structure
    expected_columns = {
        'instance_id', 'runtime_bin', 'cpu_utilization', 'memory_utilization',
        'task_count', 'cpu_capacity', 'memory_capacity', 'cpu_used', 'memory_used'
    }
    assert set(stats.columns) == expected_columns
    
    # Check stats validity
    assert all(stats['cpu_utilization'] >= 0)
    assert all(stats['cpu_utilization'] <= 1)
    assert all(stats['memory_utilization'] >= 0)
    assert all(stats['memory_utilization'] <= 1)
    assert all(stats['task_count'] >= 0)

def test_validation(packer, sample_tasks, sample_instances):
    """Test packing validation"""
    # Valid packing
    packer.pack_tasks(sample_tasks, sample_instances)
    is_valid, messages = packer.validate_packing()
    assert is_valid
    assert len(messages) == 0
    
    # Invalid packing (simulate overcommitment)
    instance = list(packer.instances.values())[0]
    instance.cpu_used = instance.cpu_capacity * 1.1  # Exceed CPU capacity
    
    is_valid, messages = packer.validate_packing()
    assert not is_valid
    assert len(messages) > 0
    assert any('CPU overcommitted' in msg for msg in messages)

def test_priority_handling(packer):
    """Test handling of task priorities"""
    # Create tasks with different priorities
    high_priority_tasks = [
        Task("task1", "job1", 50, 2.0, 4.0, 1, 2),  # High priority
        Task("task2", "job1", 50, 2.0, 4.0, 1, 2)   # High priority
    ]
    
    low_priority_tasks = [
        Task("task3", "job2", 50, 2.0, 4.0, 1, 1),  # Low priority
        Task("task4", "job2", 50, 2.0, 4.0, 1, 1)   # Low priority
    ]
    
    # Create instance with limited capacity
    instance = [Instance("i-1", 4.0, 8.0)]  # Can only fit two tasks
    
    # Pack all tasks
    assignments = packer.pack_tasks(
        low_priority_tasks + high_priority_tasks,
        instance
    )
    
    # Check that high priority tasks were packed first
    assigned_tasks = assignments.get("i-1", [])
    assert len(assigned_tasks) == 2
    assert all(task_id.startswith("task") and int(task_id[-1]) <= 2 
              for task_id in assigned_tasks)