# src/models/machine.py

import logging

class Machine:
    def __init__(self, machine_id, cpu_capacity, mem_capacity, platform_id=None, instance_type_index=0):
        self.machine_id = machine_id
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.platform_id = platform_id
        self.instance_type_index = instance_type_index  # Added attribute
        self.running_tasks = []
        self.logger = logging.getLogger(f"Machine-{self.machine_id}")
        self.logger.debug(
            f"Initialized Machine {self.machine_id} with CPU {self.cpu_capacity}, "
            f"Memory {self.mem_capacity}, Instance Type Index {self.instance_type_index}."
        )

    def can_allocate(self, cpu_req, mem_req):
        used_cpu = sum(t.cpu_req for t in self.running_tasks)
        used_mem = sum(t.mem_req for t in self.running_tasks)
        can_allocate = (used_cpu + cpu_req <= self.cpu_capacity) and (used_mem + mem_req <= self.mem_capacity)
        self.logger.debug(
            f"Can allocate CPU {cpu_req}, Memory {mem_req}: {'Yes' if can_allocate else 'No'}."
        )
        return can_allocate

    def allocate_task(self, task):
        self.running_tasks.append(task)
        self.logger.debug(
            f"Allocated Task {task.job_id}-{task.task_index} to Machine {self.machine_id}. "
            f"Current running tasks: {[t.task_index for t in self.running_tasks]}"
        )

    def remove_task(self, task):
        self.running_tasks.remove(task)
        self.logger.debug(
            f"Removed Task {task.job_id}-{task.task_index} from Machine {self.machine_id}. "
            f"Remaining running tasks: {[t.task_index for t in self.running_tasks]}"
        )
