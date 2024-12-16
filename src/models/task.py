# src/models/task.py

import logging

class Task:
    def __init__(self, job_id, task_index, submit_time, cpu_req, mem_req, disk_req, priority, runtime=None):
        self.job_id = job_id
        self.task_index = task_index
        self.submit_time = submit_time
        self.cpu_req = cpu_req
        self.mem_req = mem_req
        self.disk_req = disk_req
        self.priority = priority
        self.runtime = runtime  # Actual runtime if known (post-hoc)
        self.estimated_runtime = runtime if runtime is not None else 10.0  # Default fallback
        self.start_time = None
        self.finish_time = None
        self.machine_id = None
        self.logger = logging.getLogger(f"Task-{self.job_id}-{self.task_index}")
        self.logger.debug(f"Task initialized with CPU {self.cpu_req}, Memory {self.mem_req}, Disk {self.disk_req}, Priority {self.priority}.")
    
    def set_estimated_runtime(self, est):
        self.estimated_runtime = est
        self.logger.debug(f"Estimated runtime set to {est}.")
