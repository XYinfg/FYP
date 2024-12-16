# src/models/packer.py

import logging
import math

class Packer:
    def __init__(self, bins=[1,2,4,8,16,32,64]):
        self.bins = bins
        self.logger = logging.getLogger("Packer")
    
    def get_bin_for_runtime(self, rt):
        for b in self.bins:
            if rt < b:
                return b
        return self.bins[-1]
    
    def pack_tasks(self, tasks, machines, logger=None):
        self.logger.debug("Starting to pack tasks into machines.")
        # Group tasks by runtime bin
        bin_map = {}
        for t in tasks:
            b = self.get_bin_for_runtime(t.estimated_runtime)
            if b not in bin_map:
                bin_map[b] = []
            bin_map[b].append(t)
        
        scheduled = []
        for bin_val, task_list in bin_map.items():
            self.logger.debug(f"Packing tasks in runtime bin {bin_val} with {len(task_list)} tasks.")
            for task in task_list:
                placed = False
                for m in machines:
                    if m.can_allocate(task.cpu_req, task.mem_req):
                        m.allocate_task(task)
                        task.machine_id = m.machine_id
                        self.logger.debug(f"Task {task.job_id}-{task.task_index} placed on Machine {m.machine_id}.")
                        scheduled.append(task)
                        placed = True
                        break
                if not placed:
                    self.logger.debug(f"Task {task.job_id}-{task.task_index} could not be placed on existing machines.")
                    # Task remains unscheduled, will be handled by scaler
        self.logger.debug(f"Packer scheduled {len(scheduled)} tasks.")
        return scheduled
