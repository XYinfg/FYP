
# src/models/scaler.py

from src.models.machine import Machine
import logging

class Scaler:
    def __init__(self, instance_types):
        self.instance_types = instance_types
        self.id_count = 100000  # Starting ID for new machines
        self.utilization_history = {}  # machine_id -> utilization measurements
        self.logger = logging.getLogger("Scaler")
    
    def scale_out(self, unscheduled_tasks, logger=None):
        self.logger.debug("Starting scale-out process.")
        if not unscheduled_tasks:
            self.logger.debug("No unscheduled tasks to scale out.")
            return []
    
        # Find maximum resource requests among unscheduled tasks
        max_cpu = max(t.cpu_req for t in unscheduled_tasks)
        max_mem = max(t.mem_req for t in unscheduled_tasks)
    
        # Identify candidate instance types that can accommodate the tasks
        candidate_types = [
            (index, it) for index, it in enumerate(self.instance_types) 
            if it[0] >= max_cpu and it[1] >= max_mem
        ]
        if not candidate_types:
            self.logger.warning("No suitable instance types found for scaling out.")
            return []
        
        # Select the cheapest suitable instance type
        candidate_types.sort(key=lambda x: x[1][2])  # Sort by price
        chosen_index, chosen = candidate_types[0]
        self.logger.debug(
            f"Chosen Instance Type Index {chosen_index} with CPU {chosen[0]}, Memory {chosen[1]}, Price {chosen[2]}"
        )
        
        # Create a new Machine instance with the chosen instance type
        new_machine = Machine(
            machine_id=self.id_count, 
            cpu_capacity=chosen[0], 
            mem_capacity=chosen[1], 
            platform_id=None,
            instance_type_index=chosen_index  # Assign instance type index
        )
        self.logger.info(
            f"Scaling out: Added Machine {self.id_count} with CPU {chosen[0]}, "
            f"Memory {chosen[1]}, Price {chosen[2]}, Instance Type Index {chosen_index}."
        )
        self.id_count += 1
        return [new_machine]
    
    def scale_in(self, machines, current_time, logger=None):
        self.logger.debug("Starting scale-in process.")
        # Example scale-in logic: remove machines with no running tasks
        to_remove = []
        for m in machines:
            if not m.running_tasks:
                to_remove.append(m)
        for m in to_remove:
            machines.remove(m)
            self.logger.info(f"Scaling in: Removed Machine {m.machine_id} due to no running tasks.")
        self.logger.debug("Scale-in process completed.")
