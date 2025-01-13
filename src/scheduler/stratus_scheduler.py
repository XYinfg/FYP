import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
import uuid
import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum

from src.models.runtime_estimator import RuntimeEstimator
from src.models.packer import Packer, Task, Instance
from src.models.scaler import Scaler, InstanceType, ScalingDecision
from src.scheduler.resource_manager import ResourceManager, TaskState, TaskInfo, MachineInfo
from src.models.runtime_bins import RuntimeBinsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchedulerState(Enum):
    """Possible states for the scheduler"""
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"

@dataclass
class SchedulerConfig:
    """Configuration for the Stratus scheduler"""
    scheduling_interval: float = 10.0  # seconds
    min_instances: int = 1
    max_instances: int = 100
    runtime_bins: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period: int = 300  # seconds

class StratusScheduler:
    """
    Main scheduler component for the Stratus system.
    Coordinates resource management, task scheduling, and scaling decisions.
    """
    
    def __init__(self, config: SchedulerConfig, instance_types: List[InstanceType]):
        """
        Initialize the Stratus scheduler.
        
        Args:
            config: Scheduler configuration
            instance_types: Available instance types
        """
        self.config = config
        self.state = SchedulerState.INITIALIZED
        
        # Initialize components
        self.resource_manager = ResourceManager()
        self.runtime_bins = RuntimeBinsManager(n_bins=config.runtime_bins)
        self.runtime_estimator = RuntimeEstimator()
        self.packer = Packer(n_runtime_bins=config.runtime_bins)
        self.scaler = Scaler(
            instance_types=instance_types,
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            scale_up_threshold=config.scale_up_threshold,
            scale_down_threshold=config.scale_down_threshold,
            cooldown_period=config.cooldown_period
        )
        
        # Setup queues for events
        self.task_queue = Queue()
        self.scaling_queue = Queue()
        
        # Setup metrics tracking
        self.metrics_history: List[Dict] = []
        self.last_scheduling_time = datetime.min
        
        # Thread for main scheduling loop
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
    
    def start(self) -> None:
        """Start the scheduler"""
        if self.state != SchedulerState.INITIALIZED:
            raise RuntimeError("Scheduler can only be started from INITIALIZED state")
        
        logger.info("Starting Stratus scheduler...")
        self.state = SchedulerState.RUNNING
        self.stop_flag.clear()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduling_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Scheduler started successfully")
    
    def stop(self) -> None:
        """Stop the scheduler"""
        if self.state != SchedulerState.RUNNING:
            return
        
        logger.info("Stopping scheduler...")
        self.state = SchedulerState.STOPPED
        self.stop_flag.set()
        
        if self.scheduler_thread:
            self.scheduler_thread.join()
            self.scheduler_thread = None
        
        logger.info("Scheduler stopped")
    
    def submit_task(self,
                    job_id: str,
                    cpu_request: float,
                    memory_request: float,
                    priority: int = 1) -> str:
        """
        Submit a new task to the scheduler.
        
        Args:
            job_id: Job identifier
            cpu_request: Requested CPU resources
            memory_request: Requested memory resources
            priority: Task priority
            
        Returns:
            Task identifier
        """
        # Estimate task runtime
        # Prepare features for runtime prediction
        prediction_data = pd.DataFrame({
            'cpu_request': [cpu_request],
            'memory_request': [memory_request],
            'priority': [priority]
        })
        
        # Get runtime estimate
        runtime_estimate = self.runtime_estimator.predict(prediction_data)[0]
        
        # Determine runtime bin
        runtime_bin = self.runtime_bins.get_bin_index(runtime_estimate)
        
        # Add task to resource manager
        task_id = self.resource_manager.add_task(
            job_id=job_id,
            cpu_request=cpu_request,
            memory_request=memory_request,
            runtime_estimate=runtime_estimate,
            runtime_bin=runtime_bin,
            priority=priority
        )
        
        # Queue task for scheduling
        self.task_queue.put(task_id)
        
        logger.info(f"Submitted task {task_id} for job {job_id}")
        return task_id
    
    def add_machine(self,
                   instance_type: InstanceType,
                   machine_id: Optional[str] = None) -> str:
        """
        Add a new machine to the cluster.
        
        Args:
            instance_type: Type of instance to add
            machine_id: Optional machine identifier
            
        Returns:
            Machine identifier
        """
        machine_id = machine_id or f"m-{str(uuid.uuid4())[:8]}"
        
        self.resource_manager.add_machine(
            machine_id=machine_id,
            cpu_capacity=instance_type.cpu_capacity,
            memory_capacity=instance_type.memory_capacity,
            platform_id=instance_type.platform_id
        )
        
        self.scaler.register_instance(machine_id, instance_type.name)
        
        logger.info(f"Added machine {machine_id} of type {instance_type.name}")
        return machine_id
    
    def _scheduling_loop(self) -> None:
        """Main scheduling loop"""
        while not self.stop_flag.is_set():
            try:
                current_time = datetime.now()
                
                # Process any pending scaling decisions
                self._handle_scaling_decisions()
                
                # Only run scheduling if enough time has passed
                if (current_time - self.last_scheduling_time).total_seconds() >= self.config.scheduling_interval:
                    self._schedule_tasks()
                    self._update_metrics()
                    self.last_scheduling_time = current_time
                
                # Small sleep to prevent tight loop
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in scheduling loop: {str(e)}")
                # Continue running despite errors
                time.sleep(5)
    
    def _schedule_tasks(self) -> None:
        """Process and schedule pending tasks"""
        # Get current state
        pending_tasks = self.resource_manager.get_task_stats()
        pending_tasks = pending_tasks[pending_tasks['state'] == TaskState.PENDING.value]
        machine_stats = self.resource_manager.get_machine_stats()
        
        if pending_tasks.empty:
            return
        
        # Convert tasks and machines to packer format
        tasks = [
            Task(
                task_id=row['task_id'],
                job_id=row['job_id'],
                runtime=row['runtime_estimate'],
                cpu_request=row['cpu_request'],
                memory_request=row['memory_request'],
                runtime_bin=row['runtime_bin'],
                priority=row['priority']
            )
            for _, row in pending_tasks.iterrows()
        ]
        
        instances = [
            Instance(
                instance_id=row['machine_id'],
                cpu_capacity=row['cpu_capacity'],
                memory_capacity=row['memory_capacity'],
                cpu_used=row['cpu_used'],
                memory_used=row['memory_used'],
                runtime_bin=row['runtime_bin']
            )
            for _, row in machine_stats.iterrows()
        ]
        
        # Pack tasks onto instances
        assignments = self.packer.pack_tasks(tasks, instances)
        
        # Apply assignments
        for instance_id, task_ids in assignments.items():
            for task_id in task_ids:
                self.resource_manager.schedule_task(task_id, instance_id)
        
        # Evaluate scaling needs
        scaling_decisions = self.scaler.evaluate_scaling(pending_tasks, machine_stats)
        for decision in scaling_decisions:
            self.scaling_queue.put(decision)
    
    def _handle_scaling_decisions(self) -> None:
        """Process pending scaling decisions"""
        try:
            while True:
                decision = self.scaling_queue.get_nowait()
                
                if decision.action == 'ACQUIRE' and decision.instance_type:
                    self.add_machine(decision.instance_type)
                
                elif decision.action == 'RELEASE' and decision.instance_id:
                    affected_tasks = self.resource_manager.remove_machine(decision.instance_id)
                    self.scaler.deregister_instance(decision.instance_id)
                    
                    # Requeue affected tasks
                    for task_id in affected_tasks:
                        self.task_queue.put(task_id)
        
        except Empty:
            pass
    
    def _update_metrics(self) -> None:
        """Update scheduler metrics"""
        current_time = datetime.now()
        
        # Get current state
        machine_stats = self.resource_manager.get_machine_stats()
        task_stats = self.resource_manager.get_task_stats()
        cost_estimate = self.scaler.estimate_cost(window_hours=1.0)
        
        metrics = {
            'timestamp': current_time,
            'total_machines': len(machine_stats),
            'active_tasks': len(task_stats[task_stats['state'].isin(['SCHEDULED', 'RUNNING'])]),
            'pending_tasks': len(task_stats[task_stats['state'] == 'PENDING']),
            'completed_tasks': len(task_stats[task_stats['state'] == 'COMPLETED']),
            'failed_tasks': len(task_stats[task_stats['state'] == 'FAILED']),
            'avg_cpu_utilization': machine_stats['cpu_utilization'].mean() if not machine_stats.empty else 0.0,
            'avg_memory_utilization': machine_stats['memory_utilization'].mean() if not machine_stats.empty else 0.0,
            'estimated_hourly_cost': cost_estimate['total_cost']
        }
        
        self.metrics_history.append(metrics)
    
    def get_metrics(self) -> pd.DataFrame:
        """
        Get scheduler metrics history.
        
        Returns:
            DataFrame containing metrics history
        """
        return pd.DataFrame(self.metrics_history)
    
    def get_cost_analysis(self) -> Dict[str, float]:
        """
        Get cost analysis for the current deployment.
        
        Returns:
            Dictionary containing cost metrics
        """
        cost_estimate = self.scaler.estimate_cost(window_hours=1.0)
        task_stats = self.resource_manager.get_task_stats()
        completed_tasks = task_stats[task_stats['state'] == 'COMPLETED']
        
        if len(completed_tasks) > 0:
            avg_task_runtime = completed_tasks['actual_runtime'].mean()
            cost_per_task = cost_estimate['total_cost'] / len(completed_tasks)
        else:
            avg_task_runtime = 0.0
            cost_per_task = 0.0
        
        return {
            'hourly_cost': cost_estimate['total_cost'],
            'cost_by_instance_type': cost_estimate['cost_by_type'],
            'avg_cost_per_task': cost_per_task,
            'avg_task_runtime': avg_task_runtime
        }