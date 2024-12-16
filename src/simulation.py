import heapq
from src.utils import convert_timestamp
from src.metrics import Metrics
from src.models.task import Task
from src.models.machine import Machine
import itertools
from typing import List, Tuple
import logging

class Simulation:
    def __init__(self, tasks: List[Task], machines: List[Machine], scheduler):
        self.tasks = tasks
        self.machines = machines
        self.scheduler = scheduler
        self.current_time = 0.0
        self.event_queue: List[Tuple[float, int, str, Task]] = []
        self.metrics = Metrics()
        self.counter = itertools.count()  # Unique sequence count
        self.logger = logging.getLogger("Simulation")
    
        # Initialize tasks by submit_time
        self.pending_tasks: List[Task] = []
        self.tasks_sorted = sorted(self.tasks, key=lambda t: t.submit_time)
    
    def run(self) -> Tuple[dict, List[Machine]]:
        self.logger.debug("Starting simulation run.")
        # 1. Add SUBMIT events for tasks
        for t in self.tasks_sorted:
            self.logger.debug(
                f"Scheduling TASK_ARRIVE for Task {t.job_id}-{t.task_index} at time {t.submit_time}."
            )
            heapq.heappush(self.event_queue, (t.submit_time, next(self.counter), 'TASK_ARRIVE', t))
    
        while self.event_queue:
            timestamp, _, etype, obj = heapq.heappop(self.event_queue)
            self.current_time = timestamp
            self.logger.debug(
                f"Processing event {etype} for Task {obj.job_id}-{obj.task_index} at time {self.current_time}."
            )
            
            if etype == 'TASK_ARRIVE':
                self.pending_tasks.append(obj)
                self.logger.debug(f"Task {obj.job_id}-{obj.task_index} added to pending tasks.")
                # Attempt to schedule
                scheduled = self.scheduler.schedule(self.pending_tasks, self.machines, self.current_time)
                self.logger.debug(f"Scheduler returned {len(scheduled)} scheduled tasks.")
                # Remove scheduled tasks from pending
                for st in scheduled:
                    if st.runtime is None:
                        if st.estimated_runtime is None:
                            # Assign a default estimated runtime if none is provided
                            st.estimated_runtime = 10.0
                            self.logger.debug(
                                f"Task {st.job_id}-{st.task_index} has no estimated_runtime. Assigning default 10.0."
                            )
                        st.runtime = st.estimated_runtime
                    # Check again to ensure runtime is set
                    if st.runtime is None:
                        self.logger.error(
                            f"Task {st.job_id}-{st.task_index} still has no runtime after assignment."
                        )
                        raise ValueError(f"Task {st.job_id}-{st.task_index} has no runtime.")
                    finish_time = self.current_time + st.runtime
                    self.logger.debug(
                        f"Scheduling TASK_FINISH for Task {st.job_id}-{st.task_index} at time {finish_time}."
                    )
                    heapq.heappush(self.event_queue, (finish_time, next(self.counter), 'TASK_FINISH', st))
                    self.pending_tasks.remove(st)
    
            elif etype == 'TASK_FINISH':
                t = obj
                self.logger.debug(f"Task {t.job_id}-{t.task_index} finished at time {self.current_time}.")
                # Remove from machine
                for m in self.machines:
                    if m.machine_id == t.machine_id:
                        m.remove_task(t)
                        self.logger.debug(f"Task {t.job_id}-{t.task_index} removed from Machine {m.machine_id}.")
                        break
                # Update metrics
                self.metrics.record_task_completion(t, self.current_time)
                self.logger.debug(f"Metrics updated for Task {t.job_id}-{t.task_index}.")
                # Update runtime estimator
                if hasattr(self.scheduler, 'estimator'):
                    self.scheduler.estimator.update(t)
                    self.logger.debug(
                        f"Runtime estimator updated with Task {t.job_id}-{t.task_index} runtime."
                    )
    
        self.logger.debug("Simulation run completed.")
        # Collect machine_instance_mapping
        machine_instance_mapping = {m.machine_id: m.instance_type_index for m in self.machines}
        return self.metrics.get_results(machine_instance_mapping), self.machines