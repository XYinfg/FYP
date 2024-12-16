# src/metrics.py

import logging
from src.config import INSTANCE_TYPES

class Metrics:
    def __init__(self):
        self.completion_times = []
        self.machine_usage = {}  # machine_id -> total active time
        self.total_cost = 0.0
        self.logger = logging.getLogger("Metrics")
    
    def record_task_completion(self, task, finish_time):
        completion_time = finish_time - task.submit_time
        self.completion_times.append(completion_time)
        self.logger.debug(f"Task {task.job_id}-{task.task_index} completed in {completion_time} seconds.")
        
        # Update machine usage
        machine_id = task.machine_id
        if machine_id not in self.machine_usage:
            self.machine_usage[machine_id] = 0.0
        self.machine_usage[machine_id] += task.runtime
        self.logger.debug(f"Machine {machine_id} usage updated by {task.runtime} seconds.")
    
    def calculate_total_cost(self, machine_instance_mapping):
        # machine_instance_mapping: machine_id -> instance_type index
        for machine_id, usage_time in self.machine_usage.items():
            if machine_id in machine_instance_mapping:
                instance_type_index = machine_instance_mapping[machine_id]
                instance_type = INSTANCE_TYPES[instance_type_index]
                price = instance_type[2]
                self.total_cost += usage_time * price
                self.logger.debug(
                    f"Machine {machine_id} cost: {usage_time} * {price} = {usage_time * price}."
                )
            else:
                self.logger.warning(f"Machine {machine_id} not found in instance type mapping.")
    
    def get_results(self, machine_instance_mapping):
        # Compute aggregates
        avg_completion = sum(self.completion_times) / len(self.completion_times) if self.completion_times else 0
        self.calculate_total_cost(machine_instance_mapping)
        self.logger.debug(f"Average completion time: {avg_completion}")
        self.logger.debug(f"Total cost: {self.total_cost}")
        return {
            'avg_completion_time': avg_completion,
            'total_cost': self.total_cost
            # Add more metrics as needed
        }