import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstanceType:
    """Represents a cloud instance type configuration"""
    name: str
    cpu_capacity: float
    memory_capacity: float
    price_per_hour: float
    platform_id: str = "default"

@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    action: str  # 'ACQUIRE' or 'RELEASE'
    instance_type: Optional[InstanceType] = None
    instance_id: Optional[str] = None
    tasks_to_migrate: Optional[List[str]] = None

class Scaler:
    """
    Handles instance scaling decisions in the Stratus scheduler.
    Implements cost-aware scaling and instance selection.
    """
    
    def __init__(self, 
                 instance_types: List[InstanceType],
                 min_instances: int = 1,
                 max_instances: int = 100,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 cooldown_period: int = 300):  # 5 minutes in seconds
        """
        Initialize the Scaler.
        
        Args:
            instance_types: List of available instance types
            min_instances: Minimum number of instances to maintain
            max_instances: Maximum number of instances allowed
            scale_up_threshold: Utilization threshold to trigger scale up
            scale_down_threshold: Utilization threshold to trigger scale down
            cooldown_period: Minimum time between scaling actions (seconds)
        """
        self.instance_types = {it.name: it for it in instance_types}
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.last_scaling_time = datetime.min
        self.instance_history: List[Dict] = []
        self.current_instances: Dict[str, InstanceType] = {}
    
    def calculate_cost_efficiency(self,
                                instance_type: InstanceType,
                                task_group_cpu: float,
                                task_group_memory: float) -> float:
        """
        Calculate cost efficiency score for an instance type.
        
        Args:
            instance_type: Instance type to evaluate
            task_group_cpu: Total CPU requirements
            task_group_memory: Total memory requirements
            
        Returns:
            Cost efficiency score
        """
        # Calculate resource utilization
        cpu_util = min(task_group_cpu / instance_type.cpu_capacity, 1.0)
        memory_util = min(task_group_memory / instance_type.memory_capacity, 1.0)
        
        # Use the most constraining resource
        constraining_util = min(cpu_util, memory_util)
        
        # Calculate cost efficiency (higher is better)
        if instance_type.price_per_hour > 0:
            return constraining_util / instance_type.price_per_hour
        return 0.0
    
    def select_instance_type(self,
                           task_group_cpu: float,
                           task_group_memory: float,
                           runtime_bin: int) -> Optional[InstanceType]:
        """
        Select the most cost-efficient instance type for a group of tasks.
        
        Args:
            task_group_cpu: Total CPU requirements
            task_group_memory: Total memory requirements
            runtime_bin: Runtime bin of the task group
            
        Returns:
            Selected instance type or None if no suitable type found
        """
        best_score = -1
        best_type = None
        
        for instance_type in self.instance_types.values():
            # Check if instance can accommodate the task group
            if (task_group_cpu <= instance_type.cpu_capacity and
                task_group_memory <= instance_type.memory_capacity):
                
                score = self.calculate_cost_efficiency(
                    instance_type,
                    task_group_cpu,
                    task_group_memory
                )
                
                if score > best_score:
                    best_score = score
                    best_type = instance_type
        
        return best_type
    
    def evaluate_scaling(self,
                        pending_tasks: pd.DataFrame,
                        instance_stats: pd.DataFrame) -> List[ScalingDecision]:
        """
        Evaluate current state and make scaling decisions.
        
        Args:
            pending_tasks: DataFrame containing pending tasks
            instance_stats: DataFrame containing current instance statistics
            
        Returns:
            List of scaling decisions
        """
        try:
            current_time = datetime.now()
            decisions: List[ScalingDecision] = []
            
            # Check cooldown period
            if (current_time - self.last_scaling_time).total_seconds() < self.cooldown_period:
                return decisions
            
            # Calculate current utilization
            if not instance_stats.empty:
                avg_cpu_util = instance_stats['cpu_utilization'].mean()
                avg_memory_util = instance_stats['memory_utilization'].mean()
                overall_util = min(avg_cpu_util, avg_memory_util)
            else:
                overall_util = 0.0
            
            # Scale up if needed
            if (overall_util > self.scale_up_threshold or 
                (not instance_stats.empty and not pending_tasks.empty)):
                
                if len(self.current_instances) < self.max_instances:
                    scale_up_decision = self._evaluate_scale_up(pending_tasks)
                    if scale_up_decision:
                        decisions.append(scale_up_decision)
            
            # Scale down if needed
            elif overall_util < self.scale_down_threshold and len(self.current_instances) > self.min_instances:
                scale_down_decisions = self._evaluate_scale_down(instance_stats)
                decisions.extend(scale_down_decisions)
            
            if decisions:
                self.last_scaling_time = current_time
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error in evaluate_scaling: {str(e)}")
            return []
    
    def _evaluate_scale_up(self, pending_tasks: pd.DataFrame) -> Optional[ScalingDecision]:
        """
        Evaluate whether and how to scale up.
        
        Args:
            pending_tasks: DataFrame containing pending tasks
            
        Returns:
            Scaling decision or None
        """
        if pending_tasks.empty:
            return None
        
        # Group tasks by runtime bin
        task_groups = pending_tasks.groupby('runtime_bin')
        
        for runtime_bin, group in task_groups:
            total_cpu = group['cpu_request'].sum()
            total_memory = group['memory_request'].sum()
            
            instance_type = self.select_instance_type(
                total_cpu,
                total_memory,
                runtime_bin
            )
            
            if instance_type:
                return ScalingDecision(
                    action='ACQUIRE',
                    instance_type=instance_type
                )
        
        return None
    
    def _evaluate_scale_down(self, instance_stats: pd.DataFrame) -> List[ScalingDecision]:
        """
        Evaluate whether and how to scale down.
        
        Args:
            instance_stats: DataFrame containing instance statistics
            
        Returns:
            List of scaling decisions
        """
        decisions = []
        
        # Find underutilized instances
        underutilized = instance_stats[
            (instance_stats['cpu_utilization'] < self.scale_down_threshold) &
            (instance_stats['memory_utilization'] < self.scale_down_threshold)
        ]
        
        for _, instance in underutilized.iterrows():
            # Don't scale below minimum instances
            if len(self.current_instances) <= self.min_instances:
                break
                
            decisions.append(ScalingDecision(
                action='RELEASE',
                instance_id=instance['instance_id'],
                tasks_to_migrate=instance['task_ids'] if 'task_ids' in instance else []
            ))
        
        return decisions
    
    def register_instance(self, instance_id: str, instance_type_name: str) -> None:
        """
        Register a new instance with the scaler.
        
        Args:
            instance_id: Instance identifier
            instance_type_name: Name of the instance type
        """
        if instance_type_name not in self.instance_types:
            raise ValueError(f"Unknown instance type: {instance_type_name}")
        
        self.current_instances[instance_id] = self.instance_types[instance_type_name]
        
        self.instance_history.append({
            'timestamp': datetime.now(),
            'action': 'ACQUIRE',
            'instance_id': instance_id,
            'instance_type': instance_type_name
        })
        
        logger.info(f"Registered instance {instance_id} of type {instance_type_name}")
    
    def deregister_instance(self, instance_id: str) -> None:
        """
        Deregister an instance from the scaler.
        
        Args:
            instance_id: Instance identifier
        """
        if instance_id in self.current_instances:
            instance_type = self.current_instances[instance_id]
            del self.current_instances[instance_id]
            
            self.instance_history.append({
                'timestamp': datetime.now(),
                'action': 'RELEASE',
                'instance_id': instance_id,
                'instance_type': instance_type.name
            })
            
            logger.info(f"Deregistered instance {instance_id}")
    
    def get_instance_history(self) -> pd.DataFrame:
        """
        Get history of instance scaling actions.
        
        Returns:
            DataFrame containing scaling history
        """
        return pd.DataFrame(self.instance_history)
    
    def estimate_cost(self, window_hours: float = 1.0) -> Dict[str, float]:
        """
        Estimate cost for current instances over a time window.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Dictionary containing cost estimates
        """
        cost_by_type = defaultdict(float)
        total_cost = 0.0
        
        for instance_id, instance_type in self.current_instances.items():
            cost = instance_type.price_per_hour * window_hours
            cost_by_type[instance_type.name] += cost
            total_cost += cost
        
        return {
            'total_cost': total_cost,
            'cost_by_type': dict(cost_by_type)
        }