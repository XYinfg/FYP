# src/models/runtime_estimator.py

import logging

class RuntimeEstimator:
    def __init__(self):
        self.job_history = {}  # job_id -> list of runtimes
        self.logger = logging.getLogger("RuntimeEstimator")
    
    def update(self, task):
        jid = task.job_id
        if jid not in self.job_history:
            self.job_history[jid] = []
        self.job_history[jid].append(task.runtime)
        self.logger.debug(f"RuntimeEstimator updated with Task {task.job_id}-{task.task_index} runtime {task.runtime}.")
    
    def estimate(self, task):
        jid = task.job_id
        if jid in self.job_history and len(self.job_history[jid]) > 0:
            avg_runtime = sum(self.job_history[jid]) / len(self.job_history[jid])
            self.logger.debug(f"Estimated runtime for Task {task.job_id}-{task.task_index}: {avg_runtime} based on history.")
            return avg_runtime
        else:
            self.logger.debug(f"No history for Job {jid}. Using default runtime 10.0.")
            return 10.0  # Default estimate if no history
