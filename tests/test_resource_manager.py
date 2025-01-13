import pytest
from datetime import datetime
import pandas as pd
from src.scheduler.resource_manager import ResourceManager, TaskState

@pytest.fixture
def resource_manager():
    """Create a ResourceManager instance"""
    return ResourceManager()

@pytest.fixture
def sample_machine():
    """Sample machine configuration"""
    return {
        'machine_id': 'm-1',
        'cpu_capacity': 4.0,
        'memory_capacity': 8.0,
        'platform_id': 'platform1'
    }

@pytest.fixture
def sample_task():
    """Sample task configuration"""
    return {
        'job_id': 'job-1',
        'cpu_request': 1.0,
        'memory_request': 2.0,
        'runtime_estimate': 100.0,
        'runtime_bin': 1,
        'priority': 1
    }

def test_machine_management(resource_manager, sample_machine):
    """Test adding and removing machines"""
    # Add machine
    resource_manager.add_machine(**sample_machine)
    
    # Check machine was added
    stats = resource_manager.get_machine_stats()
    assert len(stats) == 1
    assert stats.iloc[0]['machine_id'] == sample_machine['machine_id']
    assert stats.iloc[0]['cpu_capacity'] == sample_machine['cpu_capacity']
    assert stats.iloc[0]['memory_capacity'] == sample_machine['memory_capacity']
    
    # Update machine configuration
    resource_manager.add_machine(
        machine_id=sample_machine['machine_id'],
        cpu_capacity=8.0,
        memory_capacity=16.0
    )
    
    # Check machine was updated
    stats = resource_manager.get_machine_stats()
    assert len(stats) == 1
    assert stats.iloc[0]['cpu_capacity'] == 8.0
    assert stats.iloc[0]['memory_capacity'] == 16.0
    
    # Remove machine
    affected_tasks = resource_manager.remove_machine(sample_machine['machine_id'])
    assert len(affected_tasks) == 0
    assert len(resource_manager.get_machine_stats()) == 0

def test_task_management(resource_manager, sample_machine, sample_task):
    """Test task lifecycle management"""
    # Add machine
    resource_manager.add_machine(**sample_machine)
    
    # Add task
    task_id = resource_manager.add_task(**sample_task)
    
    # Check task was added
    task_stats = resource_manager.get_task_stats()
    assert len(task_stats) == 1
    assert task_stats.iloc[0]['task_id'] == task_id
    assert task_stats.iloc[0]['state'] == TaskState.PENDING.value
    
    # Schedule task
    success = resource_manager.schedule_task(task_id, sample_machine['machine_id'])
    assert success
    
    # Check task and machine states
    task_stats = resource_manager.get_task_stats()
    machine_stats = resource_manager.get_machine_stats()
    
    assert task_stats.iloc[0]['state'] == TaskState.SCHEDULED.value
    assert task_stats.iloc[0]['machine_id'] == sample_machine['machine_id']
    assert machine_stats.iloc[0]['cpu_used'] == sample_task['cpu_request']
    assert machine_stats.iloc[0]['memory_used'] == sample_task['memory_request']
    
    # Complete task
    resource_manager.complete_task(task_id)
    
    # Check final states
    task_stats = resource_manager.get_task_stats()
    machine_stats = resource_manager.get_machine_stats()
    
    assert task_stats.iloc[0]['state'] == TaskState.COMPLETED.value
    assert machine_stats.iloc[0]['cpu_used'] == 0.0
    assert machine_stats.iloc[0]['memory_used'] == 0.0

def test_resource_constraints(resource_manager, sample_machine):
    """Test resource constraint enforcement"""
    # Add machine
    resource_manager.add_machine(**sample_machine)
    
    # Add tasks that exceed capacity
    task1_id = resource_manager.add_task(
        job_id='job-1',
        cpu_request=3.0,
        memory_request=6.0,
        runtime_estimate=100.0,
        runtime_bin=1,
        priority=1
    )
    
    task2_id = resource_manager.add_task(
        job_id='job-2',
        cpu_request=2.0,
        memory_request=4.0,
        runtime_estimate=100.0,
        runtime_bin=1,
        priority=1
    )
    
    # Schedule first task (should succeed)
    assert resource_manager.schedule_task(task1_id, sample_machine['machine_id'])
    
    # Try to schedule second task (should fail)
    assert not resource_manager.schedule_task(task2_id, sample_machine['machine_id'])
    
    # Check states
    task_stats = resource_manager.get_task_stats()
    assert task_stats[task_stats['task_id'] == task1_id].iloc[0]['state'] == TaskState.SCHEDULED.value
    assert task_stats[task_stats['task_id'] == task2_id].iloc[0]['state'] == TaskState.PENDING.value

def test_runtime_bin_tracking(resource_manager, sample_machine):
    """Test tracking of runtime bins"""
    resource_manager.add_machine(**sample_machine)
    
    # Add tasks with different runtime bins
    task1_id = resource_manager.add_task(
        job_id='job-1',
        cpu_request=1.0,
        memory_request=2.0,
        runtime_estimate=100.0,
        runtime_bin=1,
        priority=1
    )
    
    task2_id = resource_manager.add_task(
        job_id='job-2',
        cpu_request=1.0,
        memory_request=2.0,
        runtime_estimate=200.0,
        runtime_bin=2,
        priority=1
    )
    
    # Schedule tasks
    resource_manager.schedule_task(task1_id, sample_machine['machine_id'])
    resource_manager.schedule_task(task2_id, sample_machine['machine_id'])
    
    # Check machine runtime bin
    machine_stats = resource_manager.get_machine_stats()
    assert machine_stats.iloc[0]['runtime_bin'] == 2  # Should be max of task bins

def test_allocation_history(resource_manager, sample_machine, sample_task):
    """Test allocation history tracking"""
    resource_manager.add_machine(**sample_machine)
    task_id = resource_manager.add_task(**sample_task)
    
    # Schedule task
    resource_manager.schedule_task(task_id, sample_machine['machine_id'])
    
    # Complete task
    resource_manager.complete_task(task_id)
    
    # Check history
    history = resource_manager.get_allocation_history()
    assert len(history) == 2  # Schedule and complete events
    
    schedule_event = history[history['action'] == 'SCHEDULE'].iloc[0]
    complete_event = history[history['action'] == 'COMPLETE'].iloc[0]
    
    assert schedule_event['task_id'] == task_id
    assert schedule_event['machine_id'] == sample_machine['machine_id']
    assert complete_event['task_id'] == task_id

def test_state_validation(resource_manager, sample_machine, sample_task):
    """Test state validation"""
    # Initial state should be valid
    is_valid, messages = resource_manager.validate_state()
    assert is_valid
    assert len(messages) == 0
    
    # Add machine and task
    resource_manager.add_machine(**sample_machine)
    task_id = resource_manager.add_task(**sample_task)
    
    # Manually corrupt state
    task = resource_manager.tasks[task_id]
    task.state = TaskState.SCHEDULED
    task.machine_id = None
    
    # State should now be invalid
    is_valid, messages = resource_manager.validate_state()
    assert not is_valid
    assert any('no machine assignment' in msg for msg in messages)

def test_machine_removal_with_tasks(resource_manager, sample_machine):
    """Test removing a machine with running tasks"""
    resource_manager.add_machine(**sample_machine)
    
    # Add and schedule multiple tasks
    task_ids = []
    for i in range(3):
        task_id = resource_manager.add_task(
            job_id=f'job-{i}',
            cpu_request=1.0,
            memory_request=2.0,
            runtime_estimate=100.0,
            runtime_bin=1,
            priority=1
        )
        resource_manager.schedule_task(task_id, sample_machine['machine_id'])
        task_ids.append(task_id)
    
    # Remove machine
    affected_tasks = resource_manager.remove_machine(sample_machine['machine_id'])
    
    # Check tasks were affected
    assert len(affected_tasks) == 3
    assert set(affected_tasks) == set(task_ids)
    
    # Check tasks were rescheduled
    task_stats = resource_manager.get_task_stats()
    for task_id in task_ids:
        task_info = task_stats[task_stats['task_id'] == task_id].iloc[0]
        assert task_info['state'] == TaskState.PENDING.value
        assert pd.isna(task_info['machine_id'])