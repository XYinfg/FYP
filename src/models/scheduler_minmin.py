from src.models.scheduler_base import SchedulerBase
import logging

class MinMinScheduler(SchedulerBase):
    def __init__(self):
        self.logger = logging.getLogger("MinMinScheduler")
    
    def schedule(self, tasks, machines, current_time):
        self.logger.debug(f"Scheduling {len(tasks)} tasks using Min-Min.")
        scheduled = []
        # Sort tasks by their estimated runtime or use a heuristic
        tasks_sorted = sorted(tasks, key=lambda t: t.cpu_req + t.mem_req)  # Simplistic heuristic
        for task in tasks_sorted:
            candidate_machine = None
            best_finish_time = None
            for m in machines:
                if m.can_allocate(task.cpu_req, task.mem_req):
                    runtime = task.estimated_runtime if task.estimated_runtime else 10.0
                    finish_time = current_time + runtime
                    if best_finish_time is None or finish_time < best_finish_time:
                        best_finish_time = finish_time
                        candidate_machine = m
            if candidate_machine:
                candidate_machine.allocate_task(task)
                task.machine_id = candidate_machine.machine_id
                task.start_time = current_time
                self.logger.debug(f"Task {task.job_id}-{task.task_index} scheduled on Machine {candidate_machine.machine_id} with finish time {best_finish_time}.")
                scheduled.append(task)
        self.logger.debug(f"Min-Min Scheduler scheduled {len(scheduled)} tasks.")
        return scheduled