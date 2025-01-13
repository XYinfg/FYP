import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.scaler import Scaler, InstanceType, ScalingDecision

@pytest.fixture
def instance_types():
    """Create sample instance types"""
    return [
        InstanceType("small", 2.0, 4.0, 0.1),    # 2 CPU, 4 GB, $0.10/hr
        InstanceType("medium", 4.0, 8.0, 0.2),   # 4 CPU, 8 GB, $0.20/hr
        InstanceType("large", 8.0, 16.0, 0.4)    # 8 CPU, 16 GB, $0.40/hr
    ]

@pytest.fixture
def scaler(instance_types):
    """Create a Scaler instance"""
    return Scaler(
        instance_types=instance_types,
        min_instances=1,
        max_instances=10,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        cooldown_period=300
    )

@pytest.fixture
def sample_pending_tasks():
    """Create sample pending tasks"""
    return pd.DataFrame({
        'task_id': ['t1', 't2', 't3'],
        'runtime_bin': [1, 1, 2],
        'cpu_request': [1.0, 1.5, 2.0],
        'memory_request': [2.0, 3.0, 4.0]
    })

@pytest.fixture
def sample_instance_stats():
    """Create sample instance statistics"""
    return pd.DataFrame({
        'instance_id': ['i1', 'i2'],
        'cpu_utilization': [0.9, 0.2],
        'memory_utilization': [0.85, 0.25],
        'task_ids': [['t1', 't2'], ['t3']]
    })

def test_cost_efficiency_calculation(scaler, instance_types):
    """Test calculation of cost efficiency scores"""
    instance_type = instance_types[0]  # small instance
    
    # Test perfect utilization
    score = scaler.calculate_cost_efficiency(
        instance_type,
        task_group_cpu=2.0,  # matches capacity
        task_group_memory=4.0  # matches capacity
    )
    assert score == 1.0 / instance_type.price_per_hour
    
    # Test underutilization
    score = scaler.calculate_cost_efficiency(
        instance_type,
        task_group_cpu=1.0,  # 50% utilization
        task_group_memory=2.0  # 50% utilization
    )
    assert score == 0.5 / instance_type.price_per_hour
    
    # Test oversubscription (should cap at 1.0)
    score = scaler.calculate_cost_efficiency(
        instance_type,
        task_group_cpu=4.0,  # 200% utilization
        task_group_memory=8.0  # 200% utilization
    )
    assert score == 1.0 / instance_type.price_per_hour

def test_instance_type_selection(scaler, instance_types):
    """Test selection of instance types based on requirements"""
    # Small workload
    selected = scaler.select_instance_type(
        task_group_cpu=1.5,
        task_group_memory=3.0,
        runtime_bin=1
    )
    assert selected.name == "small"
    
    # Medium workload
    selected = scaler.select_instance_type(
        task_group_cpu=3.0,
        task_group_memory=6.0,
        runtime_bin=1
    )
    assert selected.name == "medium"
    
    # Large workload
    selected = scaler.select_instance_type(
        task_group_cpu=6.0,
        task_group_memory=12.0,
        runtime_bin=1
    )
    assert selected.name == "large"
    
    # Too large workload
    selected = scaler.select_instance_type(
        task_group_cpu=10.0,  # Exceeds all instance types
        task_group_memory=20.0,
        runtime_bin=1
    )
    assert selected is None

def test_scaling_evaluation(scaler, sample_pending_tasks, sample_instance_stats):
    """Test scaling decision evaluation"""
    # Register initial instances
    scaler.register_instance('i1', 'small')
    scaler.register_instance('i2', 'small')
    
    # Test scale up condition
    decisions = scaler.evaluate_scaling(sample_pending_tasks, pd.DataFrame())
    assert len(decisions) == 1
    assert decisions[0].action == 'ACQUIRE'
    assert decisions[0].instance_type is not None
    
    # Allow cooldown period to pass
    scaler.last_scaling_time = datetime.now() - timedelta(seconds=scaler.cooldown_period + 1)
    
    # Test scale down condition
    low_util_stats = pd.DataFrame({
        'instance_id': ['i1', 'i2'],
        'cpu_utilization': [0.2, 0.2],
        'memory_utilization': [0.2, 0.2],
        'task_ids': [['t1'], ['t2']]
    })
    
    decisions = scaler.evaluate_scaling(pd.DataFrame(), low_util_stats)
    assert len(decisions) == 1  # Should only scale down one due to min_instances
    assert decisions[0].action == 'RELEASE'
    assert decisions[0].instance_id is not None

def test_cooldown_period(scaler, sample_pending_tasks):
    """Test cooldown period enforcement"""
    # Register initial instance
    scaler.register_instance('i1', 'small')  # small has 2 CPU, 4GB
    
    # Create heavy pending tasks that exceed capacity
    heavy_tasks = pd.DataFrame({
        'task_id': ['t1', 't2'],
        'runtime_bin': [1, 1],
        'cpu_request': [2.0, 2.0],  # Total 4 CPU > 2 CPU capacity
        'memory_request': [4.0, 4.0]  # Total 8GB > 4GB capacity
    })
    
    # Reset cooldown
    scaler.last_scaling_time = datetime.now() - timedelta(seconds=301)
    
    # First scaling decision
    decisions1 = scaler.evaluate_scaling(heavy_tasks, pd.DataFrame())
    assert len(decisions1) > 0
    
    # Immediate second attempt
    decisions2 = scaler.evaluate_scaling(heavy_tasks, pd.DataFrame())
    assert len(decisions2) == 0  # Should be blocked by cooldown
    
    # Wait for cooldown
    scaler.last_scaling_time = datetime.now() - timedelta(seconds=301)
    decisions3 = scaler.evaluate_scaling(heavy_tasks, pd.DataFrame())
    assert len(decisions3) > 0

def test_instance_management(scaler):
    """Test instance registration and deregistration"""
    # Register instances
    scaler.register_instance('i1', 'small')
    scaler.register_instance('i2', 'medium')
    assert len(scaler.current_instances) == 2
    
    # Try to register invalid instance type
    with pytest.raises(ValueError):
        scaler.register_instance('i3', 'nonexistent')
    
    # Deregister instance
    scaler.deregister_instance('i1')
    assert len(scaler.current_instances) == 1
    assert 'i1' not in scaler.current_instances
    
    # Check history
    history = scaler.get_instance_history()
    assert len(history) == 3  # 2 registers + 1 deregister
    assert sum(history['action'] == 'ACQUIRE') == 2
    assert sum(history['action'] == 'RELEASE') == 1

def test_cost_estimation(scaler):
    """Test cost estimation calculations"""
    # Register mix of instances
    scaler.register_instance('i1', 'small')   # $0.10/hr
    scaler.register_instance('i2', 'medium')  # $0.20/hr
    scaler.register_instance('i3', 'medium')  # $0.20/hr
    
    # Estimate cost for 1 hour
    costs = scaler.estimate_cost(window_hours=1.0)
    assert costs['total_cost'] == 0.5  # 0.10 + 0.20 + 0.20
    assert costs['cost_by_type'] == {
        'small': 0.1,
        'medium': 0.4
    }
    
    # Estimate cost for 2 hours
    costs = scaler.estimate_cost(window_hours=2.0)
    assert costs['total_cost'] == 1.0  # (0.10 + 0.20 + 0.20) * 2

def test_scaling_limits(scaler, sample_pending_tasks):
    """Test minimum and maximum instance limits"""
    # Register up to minimum instances
    scaler.register_instance('i1', 'small')
    
    # Try scale down
    decisions = scaler.evaluate_scaling(
        pd.DataFrame(),
        pd.DataFrame({
            'instance_id': ['i1'],
            'cpu_utilization': [0.1],
            'memory_utilization': [0.1]
        })
    )
    assert len(decisions) == 0  # Shouldn't scale below minimum
    
    # Register up to maximum instances
    for i in range(2, 11):
        scaler.register_instance(f'i{i}', 'small')
    
    # Try scale up
    decisions = scaler.evaluate_scaling(sample_pending_tasks, pd.DataFrame())
    assert len(decisions) == 0  # Shouldn't scale above maximum

def test_scaling_with_mixed_instance_types(scaler):
    """Test scaling decisions with mixed instance types"""
    # Register mix of instance types
    scaler.register_instance('i1', 'small')    # 2 CPU, 4GB
    
    # Reset cooldown
    scaler.last_scaling_time = datetime.now() - timedelta(seconds=301)
    
    # Create tasks that exceed capacity of single small instance
    pending_tasks = pd.DataFrame({
        'task_id': ['t1', 't2'],
        'runtime_bin': [1, 1],
        'cpu_request': [2.0, 2.0],     # Total 4 CPU > 2 CPU capacity
        'memory_request': [4.0, 4.0]    # Total 8GB > 4GB capacity
    })

    decisions = scaler.evaluate_scaling(pending_tasks, pd.DataFrame())
    assert len(decisions) > 0
    assert decisions[0].action == 'ACQUIRE'

def test_scale_down_priority(scaler):
    """Test that scale down prioritizes least utilized instances"""
    # Register instances
    scaler.register_instance('i1', 'small')
    scaler.register_instance('i2', 'small')
    scaler.register_instance('i3', 'small')
    
    # Create instance stats with varying utilization
    instance_stats = pd.DataFrame({
        'instance_id': ['i1', 'i2', 'i3'],
        'cpu_utilization': [0.1, 0.25, 0.2],
        'memory_utilization': [0.15, 0.2, 0.25],
        'task_ids': [['t1'], ['t2'], ['t3']]
    })
    
    decisions = scaler.evaluate_scaling(pd.DataFrame(), instance_stats)
    assert len(decisions) > 0
    if decisions[0].action == 'RELEASE':
        assert decisions[0].instance_id == 'i1'  # Should choose least utilized