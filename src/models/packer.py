import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a task to be scheduled"""
    task_id: str
    job_id: str
    runtime: float
    cpu_request: float
    memory_request: float
    runtime_bin: int
    priority: int

@dataclass
class Instance:
    """Represents a cloud instance"""
    instance_id: str
    cpu_capacity: float
    memory_capacity: float
    cpu_used: float = 0.0
    memory_used: float = 0.0
    runtime_bin: int = 0
    tasks: List[Task] = None
    
    def __post_init__(self):
        self.tasks = self.tasks or []
    
    @property
    def cpu_available(self) -> float:
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> float:
        return self.memory_capacity - self.memory_used
    
    @property
    def utilization(self) -> float:
        cpu_util = self.cpu_used / self.cpu_capacity
        mem_util = self.memory_used / self.memory_capacity
        return min(cpu_util, mem_util)

class Packer:
    """
    Implements the Stratus packing algorithm for task placement.
    Uses runtime binning and up/down packing strategies.
    """
    
    def __init__(self, n_runtime_bins: int = 10):
        """
        Initialize the Packer.
        
        Args:
            n_runtime_bins: Number of runtime bins to use
        """
        self.n_runtime_bins = n_runtime_bins
        self.instances: Dict[str, Instance] = {}
        self.runtime_bins: List[float] = []
    
    def set_runtime_bins(self, bin_edges: List[float]) -> None:
        """
        Set the runtime bin boundaries.
        
        Args:
            bin_edges: List of bin edge values
        """
        self.runtime_bins = sorted(bin_edges)
    
    def get_runtime_bin(self, runtime: float) -> int:
        """
        Determine which runtime bin a task belongs to.
        
        Args:
            runtime: Task runtime in seconds
            
        Returns:
            Bin index
        """
        if not self.runtime_bins:
            raise ValueError("Runtime bins not initialized")
        
        return np.digitize(runtime, self.runtime_bins) - 1
    
    def can_fit_task(self, task: Task, instance: Instance) -> bool:
        """
        Check if a task can fit on an instance.
        
        Args:
            task: Task to check
            instance: Instance to check
            
        Returns:
            True if task can fit, False otherwise
        """
        return (
            instance.cpu_available >= task.cpu_request and
            instance.memory_available >= task.memory_request
        )
    
    def add_task_to_instance(self, task: Task, instance: Instance) -> None:
        """
        Add a task to an instance.
        
        Args:
            task: Task to add
            instance: Target instance
        """
        instance.cpu_used += task.cpu_request
        instance.memory_used += task.memory_request
        instance.tasks.append(task)
        
        # Update instance runtime bin if needed
        instance.runtime_bin = max(instance.runtime_bin, task.runtime_bin)
    
    def find_best_instance(self, 
                          task: Task,
                          candidates: List[Instance],
                          allow_down_packing: bool = True) -> Optional[Instance]:
        """
        Find the best instance for a task.
        
        Args:
            task: Task to place
            candidates: List of candidate instances
            allow_down_packing: Whether to allow down-packing
            
        Returns:
            Best instance or None if no suitable instance found
        """
        if not candidates:
            return None
        
        # Try same bin first
        same_bin = [
            inst for inst in candidates
            if inst.runtime_bin == task.runtime_bin and self.can_fit_task(task, inst)
        ]
        if same_bin:
            # Choose instance with best utilization
            return max(same_bin, key=lambda x: x.utilization)
        
        # Try up-packing
        larger_bins = [
            inst for inst in candidates
            if inst.runtime_bin > task.runtime_bin and self.can_fit_task(task, inst)
        ]
        if larger_bins:
            # Choose instance with closest runtime bin
            return min(larger_bins, key=lambda x: x.runtime_bin)
        
        # Try down-packing if allowed
        if allow_down_packing:
            smaller_bins = [
                inst for inst in candidates
                if inst.runtime_bin < task.runtime_bin and self.can_fit_task(task, inst)
            ]
            if smaller_bins:
                # Choose instance with largest runtime bin
                return max(smaller_bins, key=lambda x: x.runtime_bin)
        
        return None
    
    def pack_tasks(self, 
                  tasks: List[Task],
                  instances: List[Instance]) -> Dict[str, List[str]]:
        """
        Pack tasks onto instances using the Stratus algorithm.
        
        Args:
            tasks: List of tasks to pack
            instances: List of available instances
            
        Returns:
            Dictionary mapping instance IDs to lists of task IDs
        """
        # Sort tasks by runtime bin (descending) and priority
        sorted_tasks = sorted(
            tasks,
            key=lambda x: (-x.runtime_bin, -x.priority)
        )
        
        # Initialize instances
        self.instances = {inst.instance_id: inst for inst in instances}
        
        # Track assignments
        assignments: Dict[str, List[str]] = defaultdict(list)
        unassigned_tasks: List[Task] = []
        
        # First pass: try normal packing
        for task in sorted_tasks:
            instance = self.find_best_instance(
                task,
                list(self.instances.values()),
                allow_down_packing=False
            )
            
            if instance:
                self.add_task_to_instance(task, instance)
                assignments[instance.instance_id].append(task.task_id)
            else:
                unassigned_tasks.append(task)
        
        # Second pass: try down-packing for remaining tasks
        for task in unassigned_tasks:
            instance = self.find_best_instance(
                task,
                list(self.instances.values()),
                allow_down_packing=True
            )
            
            if instance:
                self.add_task_to_instance(task, instance)
                assignments[instance.instance_id].append(task.task_id)
        
        return dict(assignments)
    
    def get_instance_stats(self) -> pd.DataFrame:
        """
        Get statistics about current instance utilization.
        
        Returns:
            DataFrame with instance statistics
        """
        stats = []
        for instance in self.instances.values():
            stats.append({
                'instance_id': instance.instance_id,
                'runtime_bin': instance.runtime_bin,
                'cpu_utilization': instance.cpu_used / instance.cpu_capacity,
                'memory_utilization': instance.memory_used / instance.memory_capacity,
                'task_count': len(instance.tasks),
                'cpu_capacity': instance.cpu_capacity,
                'memory_capacity': instance.memory_capacity,
                'cpu_used': instance.cpu_used,
                'memory_used': instance.memory_used
            })
        
        return pd.DataFrame(stats)
    
    def validate_packing(self) -> Tuple[bool, List[str]]:
        """
        Validate the current packing solution.
        
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        for instance in self.instances.values():
            # Check resource limits
            if instance.cpu_used > instance.cpu_capacity:
                messages.append(
                    f"Instance {instance.instance_id} CPU overcommitted: "
                    f"{instance.cpu_used:.2f} > {instance.cpu_capacity:.2f}"
                )
                is_valid = False
            
            if instance.memory_used > instance.memory_capacity:
                messages.append(
                    f"Instance {instance.instance_id} memory overcommitted: "
                    f"{instance.memory_used:.2f} > {instance.memory_capacity:.2f}"
                )
                is_valid = False
            
            # Check runtime bin consistency
            task_bins = [task.runtime_bin for task in instance.tasks]
            if task_bins:
                max_task_bin = max(task_bins)
                if instance.runtime_bin != max_task_bin:
                    messages.append(
                        f"Instance {instance.instance_id} runtime bin inconsistent: "
                        f"{instance.runtime_bin} != {max_task_bin}"
                    )
                    is_valid = False
        
        return is_valid, messages