class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.tasks = []
        self.submit_time = None

    def add_task(self, task):
        self.tasks.append(task)
        if self.submit_time is None or task.submit_time < self.submit_time:
            self.submit_time = task.submit_time
