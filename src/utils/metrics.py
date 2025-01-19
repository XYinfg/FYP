import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SchedulerMetrics:
    """
    Computes and tracks performance metrics for scheduler evaluation.
    """
    
    def __init__(self):
        """Initialize metrics tracking"""
        self.metrics_history: List[Dict] = []
        self.task_history: Dict[str, Dict] = {}
        
    def record_task_event(self,
                         task_id: str,
                         event_type: str,
                         timestamp: datetime,
                         task_info: Dict) -> None:
        """
        Record a task lifecycle event.
        
        Args:
            task_id: Task identifier
            event_type: Type of event (submit, schedule, complete, fail)
            timestamp: Event timestamp
            task_info: Additional task information
        """
        if task_id not in self.task_history:
            self.task_history[task_id] = {
                'submit_time': None,
                'schedule_time': None,
                'completion_time': None,
                'status': None,
                'waiting_time': None,
                'execution_time': None,
                'info': task_info
            }
        
        task = self.task_history[task_id]
        
        if event_type == 'submit':
            task['submit_time'] = timestamp
        elif event_type == 'schedule':
            task['schedule_time'] = timestamp
            if task['submit_time']:
                task['waiting_time'] = (timestamp - task['submit_time']).total_seconds()
        elif event_type in ['complete', 'fail']:
            task['completion_time'] = timestamp
            task['status'] = event_type
            if task['schedule_time']:
                task['execution_time'] = (timestamp - task['schedule_time']).total_seconds()
    
    def record_metrics(self,
                      timestamp: datetime,
                      active_tasks: int,
                      pending_tasks: int,
                      completed_tasks: int,
                      failed_tasks: int,
                      total_machines: int,
                      avg_cpu_util: float,
                      avg_memory_util: float,
                      cost: float) -> None:
        """
        Record point-in-time scheduler metrics.
        
        Args:
            timestamp: Current timestamp
            active_tasks: Number of running tasks
            pending_tasks: Number of waiting tasks
            completed_tasks: Number of completed tasks
            failed_tasks: Number of failed tasks
            total_machines: Number of active machines
            avg_cpu_util: Average CPU utilization
            avg_memory_util: Average memory utilization
            cost: Current cost
        """
        self.metrics_history.append({
            'timestamp': timestamp,
            'active_tasks': active_tasks,
            'pending_tasks': pending_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'total_machines': total_machines,
            'avg_cpu_util': avg_cpu_util,
            'avg_memory_util': avg_memory_util,
            'cost': cost
        })
    
    def compute_task_statistics(self) -> Dict[str, float]:
        """
        Compute statistics about task execution.
        
        Returns:
            Dictionary containing task statistics
        """
        waiting_times = []
        execution_times = []
        completed_tasks = 0
        failed_tasks = 0
        total_tasks = len(self.task_history)
        
        for task in self.task_history.values():
            if task['waiting_time'] is not None:
                waiting_times.append(task['waiting_time'])
            if task['execution_time'] is not None:
                execution_times.append(task['execution_time'])
            if task['status'] == 'complete':
                completed_tasks += 1
            elif task['status'] == 'fail':
                failed_tasks += 1
        
        return {
            'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'median_waiting_time': np.median(waiting_times) if waiting_times else 0,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'median_execution_time': np.median(execution_times) if execution_times else 0,
            'completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
            'failure_rate': failed_tasks / total_tasks if total_tasks > 0 else 0
        }
    
    def compute_resource_statistics(self) -> Dict[str, float]:
        """
        Compute statistics about resource utilization.
        
        Returns:
            Dictionary containing resource statistics
        """
        metrics_df = pd.DataFrame(self.metrics_history)
        if metrics_df.empty:
            return {}
        
        return {
            'avg_machines': metrics_df['total_machines'].mean(),
            'max_machines': metrics_df['total_machines'].max(),
            'avg_cpu_util': metrics_df['avg_cpu_util'].mean(),
            'avg_memory_util': metrics_df['avg_memory_util'].mean(),
            'total_cost': metrics_df['cost'].sum()
        }
    
    def compute_sla_metrics(self, sla_waiting_time: float = 300) -> Dict[str, float]:
        """
        Compute SLA-related metrics.
        
        Args:
            sla_waiting_time: Maximum acceptable waiting time in seconds
            
        Returns:
            Dictionary containing SLA metrics
        """
        waiting_times = [
            task['waiting_time'] for task in self.task_history.values()
            if task['waiting_time'] is not None
        ]
        
        if not waiting_times:
            return {
                'sla_violation_rate': 0,
                'avg_sla_excess': 0
            }
        
        violations = [t for t in waiting_times if t > sla_waiting_time]
        violation_rate = len(violations) / len(waiting_times)
        
        excess_times = [t - sla_waiting_time for t in violations]
        avg_excess = np.mean(excess_times) if excess_times else 0
        
        return {
            'sla_violation_rate': violation_rate,
            'avg_sla_excess': avg_excess
        }
    
    def compute_performance_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Get earliest submit time and latest completion time
        submit_times = [task['submit_time'] for task in self.task_history.values() 
                       if task['submit_time'] is not None]
        completion_times = [task['completion_time'] for task in self.task_history.values() 
                          if task['completion_time'] is not None]
        
        if not submit_times or not completion_times:
            return {
                'makespan': 0,
                'avg_turnaround_time': 0,
                'throughput': 0,
                'resource_utilization': 0,
                'scheduling_efficiency': 0,
                'completion_rate': 0
            }
        
        # Calculate makespan (total execution time)
        makespan = (max(completion_times) - min(submit_times)).total_seconds()
        
        # Calculate throughput (tasks completed per second)
        completed_tasks = len([t for t in self.task_history.values() if t['status'] == 'complete'])
        throughput = completed_tasks / makespan if makespan > 0 else 0
        
        # Calculate average turnaround time
        turnaround_times = []
        for task in self.task_history.values():
            if task['submit_time'] and task['completion_time']:
                turnaround_time = (task['completion_time'] - task['submit_time']).total_seconds()
                turnaround_times.append(turnaround_time)
        avg_turnaround_time = np.mean(turnaround_times) if turnaround_times else 0
        
        # Calculate resource utilization (time running tasks / total time available)
        total_task_time = sum(t['execution_time'] for t in self.task_history.values() 
                            if t['execution_time'] is not None)
        total_available_time = makespan * len(self.get_unique_machines())
        resource_utilization = total_task_time / total_available_time if total_available_time > 0 else 0
        
        # Calculate scheduling efficiency (1 - average waiting time / makespan)
        avg_waiting_time = np.mean([task['waiting_time'] for task in self.task_history.values() 
                                  if task['waiting_time'] is not None]) or 0
        scheduling_efficiency = 1 - (avg_waiting_time / makespan) if makespan > 0 else 0
        
        # Calculate completion rate
        total_tasks = len(self.task_history)
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        return {
            'makespan': makespan,
            'avg_turnaround_time': avg_turnaround_time,
            'throughput': throughput,
            'resource_utilization': resource_utilization,
            'scheduling_efficiency': scheduling_efficiency,
            'completion_rate': completion_rate
        }
        
    def get_unique_machines(self) -> set:
        """Get set of unique machines used"""
        machines = set()
        for task in self.task_history.values():
            if 'machine_id' in task.get('info', {}):
                machines.add(task['info']['machine_id'])
        return machines
        
    def get_full_report(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing all metrics
        """
        basic_stats = self.compute_task_statistics()
        resource_stats = self.compute_resource_statistics()
        sla_metrics = self.compute_sla_metrics()
        performance_metrics = self.compute_performance_metrics()
        
        # Calculate additional metrics
        job_count = len(set(task['info']['job_id'] for task in self.task_history.values() 
                          if 'job_id' in task.get('info', {})))
        
        system_metrics = {
            'total_jobs': job_count,
            'total_tasks': len(self.task_history),
            'unique_machines': len(self.get_unique_machines())
        }
        
        return {
            'task_stats': basic_stats,
            'resource_stats': resource_stats,
            'sla_metrics': sla_metrics,
            'performance_metrics': performance_metrics,
            'system_metrics': system_metrics
        }
    
    def save_metrics(self, output_dir: str) -> None:
        """
        Save metrics to files.
        
        Args:
            output_dir: Directory to save metric files
        """
        # Save metrics history
        pd.DataFrame(self.metrics_history).to_csv(
            f"{output_dir}/metrics_history.csv",
            index=False
        )
        
        # Save task history
        task_df = pd.DataFrame.from_dict(self.task_history, orient='index')
        task_df.to_csv(f"{output_dir}/task_history.csv")
        
        # Save summary report
        report = self.get_full_report()
        pd.DataFrame.from_dict(report).to_csv(
            f"{output_dir}/performance_report.csv"
        )