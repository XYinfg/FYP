import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict, deque
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskState(Enum):
    """Possible states for a task"""
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    KILLED = "KILLED"

@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    job_id: str
    cpu_request: float
    memory_request: float
    runtime_estimate: float
    runtime_bin: int
    priority: int
    state: TaskState = TaskState.PENDING
    machine_id: Optional[str] = None
    start_time: Optional[datetime] = None
    completion_ratio: Optional[float] = None
    actual_runtime: Optional[float] = None

@dataclass
class MachineInfo:
    """Information about a machine"""
    machine_id: str
    cpu_capacity: float
    memory_capacity: float
    cpu_used: float = 0.0
    memory_used: float = 0.0
    state: str = "ACTIVE"
    tasks: Set[str] = field(default_factory=set)
    platform_id: Optional[str] = None
    runtime_bin: int = 0

class ResourceManager:
    """
    Manages resources and task scheduling in the Stratus system.
    Tracks machine states and handles task queuing and dispatch.
    """
    
    def __init__(self):
        """Initialize the Resource Manager"""
        self.machines: Dict[str, MachineInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.pending_tasks: deque = deque()  # Tasks waiting for scheduling
        self.running_tasks: Set[str] = set()  # Currently running tasks
        self.completed_tasks: Set[str] = set()  # Successfully completed tasks
        self.failed_tasks: Set[str] = set()  # Failed or killed tasks
        
        # Track resource allocation history
        self.allocation_history: List[Dict] = []
        
    def add_machine(self, 
                   machine_id: str,
                   cpu_capacity: float,
                   memory_capacity: float,
                   platform_id: Optional[str] = None) -> None:
        """
        Add a new machine to the resource pool.
        
        Args:
            machine_id: Unique identifier for the machine
            cpu_capacity: Available CPU capacity
            memory_capacity: Available memory capacity
            platform_id: Optional platform identifier
        """
        if machine_id in self.machines:
            logger.warning(f"Machine {machine_id} already exists, updating its configuration")
            self.machines[machine_id].cpu_capacity = cpu_capacity
            self.machines[machine_id].memory_capacity = memory_capacity
            self.machines[machine_id].platform_id = platform_id
        else:
            self.machines[machine_id] = MachineInfo(
                machine_id=machine_id,
                cpu_capacity=cpu_capacity,
                memory_capacity=memory_capacity,
                platform_id=platform_id
            )
            logger.info(f"Added machine {machine_id} with {cpu_capacity} CPU, {memory_capacity} memory")

    def remove_machine(self, machine_id: str) -> List[str]:
        """
        Remove a machine from the resource pool.
        
        Args:
            machine_id: Identifier of machine to remove
            
        Returns:
            List of task IDs that were running on the machine
        """
        if machine_id not in self.machines:
            logger.warning(f"Attempted to remove non-existent machine {machine_id}")
            return []
        
        machine = self.machines[machine_id]
        affected_tasks = list(machine.tasks)
        
        # Update task states
        for task_id in affected_tasks:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.state = TaskState.PENDING
                task.machine_id = None
                self.pending_tasks.append(task_id)
                if task_id in self.running_tasks:
                    self.running_tasks.remove(task_id)
        
        # Remove the machine
        del self.machines[machine_id]
        logger.info(f"Removed machine {machine_id}, affecting {len(affected_tasks)} tasks")
        
        return affected_tasks

    def add_task(self,
                 job_id: str,
                 cpu_request: float,
                 memory_request: float,
                 runtime_estimate: float,
                 runtime_bin: int,
                 priority: int,
                 task_id: Optional[str] = None) -> str:
        """
        Add a new task to be scheduled.
        
        Args:
            job_id: Job identifier
            cpu_request: Requested CPU resources
            memory_request: Requested memory resources
            runtime_estimate: Estimated runtime in seconds
            runtime_bin: Runtime bin index
            priority: Task priority
            task_id: Optional task identifier
            
        Returns:
            Task identifier
        """
        task_id = task_id or str(uuid.uuid4())
        
        task = TaskInfo(
            task_id=task_id,
            job_id=job_id,
            cpu_request=cpu_request,
            memory_request=memory_request,
            runtime_estimate=runtime_estimate,
            runtime_bin=runtime_bin,
            priority=priority
        )
        
        self.tasks[task_id] = task
        self.pending_tasks.append(task_id)
        
        logger.info(f"Added task {task_id} for job {job_id}")
        return task_id

    def schedule_task(self, task_id: str, machine_id: str) -> bool:
        """
        Schedule a task on a specific machine.
        
        Args:
            task_id: Task identifier
            machine_id: Target machine identifier
            
        Returns:
            True if scheduling was successful
        """
        if task_id not in self.tasks or machine_id not in self.machines:
            return False
        
        task = self.tasks[task_id]
        machine = self.machines[machine_id]
        
        # Check resource availability
        if (machine.cpu_used + task.cpu_request > machine.cpu_capacity or
            machine.memory_used + task.memory_request > machine.memory_capacity):
            return False
        
        # Update task state
        task.state = TaskState.SCHEDULED
        task.machine_id = machine_id
        task.start_time = datetime.now()
        
        # Update machine state
        machine.cpu_used += task.cpu_request
        machine.memory_used += task.memory_request
        machine.tasks.add(task_id)
        machine.runtime_bin = max(machine.runtime_bin, task.runtime_bin)
        
        # Update tracking sets
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        self.running_tasks.add(task_id)
        
        # Record allocation
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'task_id': task_id,
            'machine_id': machine_id,
            'cpu_request': task.cpu_request,
            'memory_request': task.memory_request,
            'action': 'SCHEDULE'
        })
        
        logger.info(f"Scheduled task {task_id} on machine {machine_id}")
        return True

    def complete_task(self, task_id: str, success: bool = True) -> None:
        """
        Mark a task as completed or failed.
        
        Args:
            task_id: Task identifier
            success: Whether the task completed successfully
        """
        if task_id not in self.tasks:
            logger.warning(f"Attempted to complete non-existent task {task_id}")
            return
        
        task = self.tasks[task_id]
        if task.machine_id:
            machine = self.machines[task.machine_id]
            
            # Release resources
            machine.cpu_used -= task.cpu_request
            machine.memory_used -= task.memory_request
            machine.tasks.remove(task_id)
            
            # Update machine runtime bin
            if machine.tasks:
                machine.runtime_bin = max(
                    self.tasks[t].runtime_bin for t in machine.tasks
                )
            else:
                machine.runtime_bin = 0
        
        # Update task state
        task.state = TaskState.COMPLETED if success else TaskState.FAILED
        if task.start_time:
            task.actual_runtime = (datetime.now() - task.start_time).total_seconds()
        
        # Update tracking sets
        self.running_tasks.remove(task_id)
        if success:
            self.completed_tasks.add(task_id)
        else:
            self.failed_tasks.add(task_id)
        
        # Record completion
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'task_id': task_id,
            'machine_id': task.machine_id,
            'cpu_request': task.cpu_request,
            'memory_request': task.memory_request,
            'action': 'COMPLETE' if success else 'FAIL'
        })
        
        logger.info(f"Task {task_id} {'completed' if success else 'failed'}")

    def get_machine_stats(self) -> pd.DataFrame:
        """
        Get current statistics for all machines.
        
        Returns:
            DataFrame containing machine statistics
        """
        stats = []
        for machine in self.machines.values():
            stats.append({
                'machine_id': machine.machine_id,
                'cpu_capacity': machine.cpu_capacity,
                'memory_capacity': machine.memory_capacity,
                'cpu_used': machine.cpu_used,
                'memory_used': machine.memory_used,
                'cpu_utilization': machine.cpu_used / machine.cpu_capacity,
                'memory_utilization': machine.memory_used / machine.memory_capacity,
                'task_count': len(machine.tasks),
                'runtime_bin': machine.runtime_bin,
                'state': machine.state
            })
        return pd.DataFrame(stats)

    def get_task_stats(self) -> pd.DataFrame:
        """
        Get current statistics for all tasks.
        
        Returns:
            DataFrame containing task statistics
        """
        stats = []
        for task in self.tasks.values():
            stats.append({
                'task_id': task.task_id,
                'job_id': task.job_id,
                'state': task.state.value,
                'machine_id': task.machine_id,
                'cpu_request': task.cpu_request,
                'memory_request': task.memory_request,
                'runtime_estimate': task.runtime_estimate,
                'actual_runtime': task.actual_runtime,
                'runtime_bin': task.runtime_bin,
                'priority': task.priority
            })
        return pd.DataFrame(stats)

    def get_allocation_history(self) -> pd.DataFrame:
        """
        Get history of resource allocations.
        
        Returns:
            DataFrame containing allocation history
        """
        return pd.DataFrame(self.allocation_history)

    def validate_state(self) -> Tuple[bool, List[str]]:
        """
        Validate the current state of the resource manager.
        
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Check machine resource consistency
        for machine in self.machines.values():
            if machine.cpu_used > machine.cpu_capacity:
                messages.append(
                    f"Machine {machine.machine_id} CPU overcommitted: "
                    f"{machine.cpu_used:.2f} > {machine.cpu_capacity:.2f}"
                )
                is_valid = False
            
            if machine.memory_used > machine.memory_capacity:
                messages.append(
                    f"Machine {machine.machine_id} memory overcommitted: "
                    f"{machine.memory_used:.2f} > {machine.memory_capacity:.2f}"
                )
                is_valid = False
        
        # Check task state consistency
        task_states = defaultdict(set)
        for task_id, task in self.tasks.items():
            task_states[task.state].add(task_id)
            
            if task.state in (TaskState.SCHEDULED, TaskState.RUNNING):
                if not task.machine_id:
                    messages.append(
                        f"Task {task_id} is {task.state.value} but has no machine assignment"
                    )
                    is_valid = False
                elif task.machine_id not in self.machines:
                    messages.append(
                        f"Task {task_id} assigned to non-existent machine {task.machine_id}"
                    )
                    is_valid = False
        
        # Check set consistency
        if task_states[TaskState.PENDING] != set(self.pending_tasks):
            messages.append("Pending tasks set inconsistent with task states")
            is_valid = False
        
        if task_states[TaskState.RUNNING] != self.running_tasks:
            messages.append("Running tasks set inconsistent with task states")
            is_valid = False
        
        if task_states[TaskState.COMPLETED] != self.completed_tasks:
            messages.append("Completed tasks set inconsistent with task states")
            is_valid = False
        
        return is_valid, messages