from src.models.scheduler_base import SchedulerBase
from src.models.packer import Packer
from src.models.scaler import Scaler
from src.models.runtime_estimator import RuntimeEstimator
import logging
from src.config import INSTANCE_TYPES


class StratusScheduler(SchedulerBase):
    def __init__(self, instance_types):
        self.logger = logging.getLogger("StratusScheduler")
        self.packer = Packer()
        self.scaler = Scaler(instance_types)
        self.estimator = RuntimeEstimator()
    
    def schedule(self, tasks, machines, current_time):
        self.logger.debug(f"Scheduling {len(tasks)} tasks using Stratus.")
        # 1. Estimate runtime for new tasks
        for t in tasks:
            if t.estimated_runtime is None:
                t.set_estimated_runtime(self.estimator.estimate(t))
                self.logger.debug(f"Estimated runtime for Task {t.job_id}-{t.task_index}: {t.estimated_runtime}.")
    
        # 2. Attempt to pack tasks on existing machines
        scheduled = self.packer.pack_tasks(tasks, machines, self.logger)
    
        # If some tasks remain unscheduled, consider scaling out
        unscheduled = [t for t in tasks if t.machine_id is None]
        if unscheduled:
            self.logger.debug(f"{len(unscheduled)} tasks unscheduled. Scaling out.")
            new_machines = self.scaler.scale_out(unscheduled, self.logger)
            machines.extend(new_machines)
            self.logger.info(f"Scaled out {len(new_machines)} new machines.")
            # Retry packing for unscheduled tasks
            newly_scheduled = self.packer.pack_tasks(unscheduled, new_machines, self.logger)
            scheduled.extend(newly_scheduled)
    
        # Scale in if needed (cleanup)
        self.scaler.scale_in(machines, current_time, self.logger)
    
        self.logger.debug(f"Stratus Scheduler scheduled a total of {len(scheduled)} tasks.")
        return scheduled
