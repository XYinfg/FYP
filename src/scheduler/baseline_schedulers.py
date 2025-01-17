from typing import Dict, List, Optional
from datetime import datetime
import logging
import uuid
from queue import Queue

from src.scheduler.stratus_scheduler import SchedulerConfig
from src.models.scaler import InstanceType
from src.scheduler.resource_manager import ResourceManager
from src.utils.metrics import SchedulerMetrics

logger = logging.getLogger(__name__)

class BaselineScheduler:
    """Base class for baseline schedulers"""
    
    def __init__(self, config: SchedulerConfig, instance_types: List[InstanceType]):
        """
        Initialize baseline scheduler.
        
        Args:
            config: Scheduler configuration
            instance_types: Available instance types
        """
        self.config = config
        self.instance_types = {it.name: it for it in instance_types}
        self.resource_manager = ResourceManager()
        self.task_queue = Queue()
        self.metrics = SchedulerMetrics()
        
        # Always use small instances for simplicity
        self.default_instance = instance_types[0]
    
    def add_machine(self) -> str:
        """Add a new machine of default type"""
        machine_id = f"m-{str(uuid.uuid4())[:8]}"
        self.resource_manager.add_machine(
            machine_id=machine_id,
            cpu_capacity=self.default_instance.cpu_capacity,
            memory_capacity=self.default_instance.memory_capacity
        )
        return machine_id
    
    def submit_task(self,
                   job_id: str,
                   cpu_request: float,
                   memory_request: float,
                   priority: int = 1) -> str:
        """Submit a task to be scheduled"""
        task_id = str(uuid.uuid4())
        
        # Add task to resource manager
        self.resource_manager.add_task(
            job_id=job_id,
            task_id=task_id,
            cpu_request=cpu_request,
            memory_request=memory_request,
            runtime_estimate=0,  # Baseline schedulers don't use runtime estimates
            runtime_bin=0,
            priority=priority
        )
        
        # Add to queue
        self.task_queue.put(task_id)
        
        # Record metrics
        self.metrics.record_task_event(
            task_id=task_id,
            event_type='submit',
            timestamp=datetime.now(),
            task_info={
                'job_id': job_id,
                'cpu_request': cpu_request,
                'memory_request': memory_request,
                'priority': priority
            }
        )
        
        return task_id

class FCFSScheduler(BaselineScheduler):
    """
    First-Come-First-Served (FCFS) scheduler.
    Schedules tasks in order of arrival on the first available machine.
    """
    
    def schedule_tasks(self) -> None:
        """Process queued tasks"""
        while not self.task_queue.empty():
            task_id = self.task_queue.get()
            task = self.resource_manager.tasks.get(task_id)
            
            if not task:
                continue
            
            # Find first machine with enough capacity
            scheduled = False
            for machine in self.resource_manager.machines.values():
                if (machine.cpu_used + task.cpu_request <= machine.cpu_capacity and
                    machine.memory_used + task.memory_request <= machine.memory_capacity):
                    
                    success = self.resource_manager.schedule_task(task_id, machine.machine_id)
                    if success:
                        scheduled = True
                        self.metrics.record_task_event(
                            task_id=task_id,
                            event_type='schedule',
                            timestamp=datetime.now(),
                            task_info={'machine_id': machine.machine_id}
                        )
                        break
            
            # If no suitable machine found, add a new one
            if not scheduled and len(self.resource_manager.machines) < self.config.max_instances:
                machine_id = self.add_machine()
                success = self.resource_manager.schedule_task(task_id, machine_id)
                if success:
                    self.metrics.record_task_event(
                        task_id=task_id,
                        event_type='schedule',
                        timestamp=datetime.now(),
                        task_info={'machine_id': machine_id}
                    )
            
            # If still not scheduled, requeue
            if not scheduled:
                self.task_queue.put(task_id)

class BestFitScheduler(BaselineScheduler):
    """
    Best-Fit scheduler.
    Schedules tasks on machines that will have the least remaining capacity.
    """
    
    def calculate_fit_score(self, 
                          machine,
                          cpu_request: float,
                          memory_request: float) -> float:
        """
        Calculate how well a task fits on a machine.
        Lower score means better fit.
        """
        # Check if machine has enough capacity
        if (machine.cpu_used + cpu_request > machine.cpu_capacity or
            machine.memory_used + memory_request > machine.memory_capacity):
            return float('inf')
        
        # Calculate remaining capacity after placing task
        cpu_remaining = machine.cpu_capacity - (machine.cpu_used + cpu_request)
        memory_remaining = machine.memory_capacity - (machine.memory_used + memory_request)
        
        # Normalize by capacity
        cpu_score = cpu_remaining / machine.cpu_capacity
        memory_score = memory_remaining / machine.memory_capacity
        
        # Return average normalized remaining capacity (lower is better)
        return (cpu_score + memory_score) / 2
    
    def find_best_fit(self,
                     cpu_request: float,
                     memory_request: float) -> Optional[str]:
        """Find the best machine for a task"""
        best_score = float('inf')
        best_machine_id = None
        
        for machine in self.resource_manager.machines.values():
            score = self.calculate_fit_score(machine, cpu_request, memory_request)
            if score < best_score:
                best_score = score
                best_machine_id = machine.machine_id
        
        return best_machine_id
    
    def schedule_tasks(self) -> None:
        """Process queued tasks"""
        while not self.task_queue.empty():
            task_id = self.task_queue.get()
            task = self.resource_manager.tasks.get(task_id)
            
            if not task:
                continue
            
            # Find best fitting machine
            machine_id = self.find_best_fit(task.cpu_request, task.memory_request)
            
            # If no suitable machine found, add a new one if possible
            if not machine_id and len(self.resource_manager.machines) < self.config.max_instances:
                machine_id = self.add_machine()
            
            # Try to schedule the task
            if machine_id:
                success = self.resource_manager.schedule_task(task_id, machine_id)
                if success:
                    self.metrics.record_task_event(
                        task_id=task_id,
                        event_type='schedule',
                        timestamp=datetime.now(),
                        task_info={'machine_id': machine_id}
                    )
                else:
                    self.task_queue.put(task_id)
            else:
                self.task_queue.put(task_id)