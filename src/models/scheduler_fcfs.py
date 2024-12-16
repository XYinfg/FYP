from src.models.scheduler_base import SchedulerBase
import logging

class FCFSScheduler(SchedulerBase):
    def __init__(self):
        self.logger = logging.getLogger("FCFSScheduler")
    
    def schedule(self, tasks, machines, current_time):
        self.logger.debug(f"Scheduling {len(tasks)} tasks using FCFS.")
        # tasks: a list of pending tasks
        # machines: a list of available machines
        # Try to schedule each task in arrival order
        scheduled = []
        for task in tasks:
            for m in machines:
                if m.can_allocate(task.cpu_req, task.mem_req):
                    m.allocate_task(task)
                    task.machine_id = m.machine_id
                    task.start_time = current_time
                    self.logger.debug(f"Task {task.job_id}-{task.task_index} scheduled on Machine {m.machine_id}.")
                    scheduled.append(task)
                    break
        self.logger.debug(f"FCFS Scheduler scheduled {len(scheduled)} tasks.")
        return scheduled