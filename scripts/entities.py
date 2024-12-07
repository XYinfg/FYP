class Task:
    def __init__(self, job_id, task_index, arrival_time, requested_cpu, requested_ram, runtime=100):
        self.job_id = job_id
        self.task_index = task_index
        self.task_id = f"{job_id}-{task_index}"
        self.arrival_time = arrival_time
        self.requested_cpu = requested_cpu
        self.requested_ram = requested_ram
        self.estimated_runtime = runtime
        self.start_time = None
        self.finish_time = None

class Machine:
    def __init__(self, machine_id, capacity_cpu, capacity_mem, cost_per_hour=0.1):
        self.machine_id = machine_id
        self.capacity_cpu = capacity_cpu
        self.capacity_mem = capacity_mem
        self.cost_per_hour = cost_per_hour
        self.used_cpu = 0.0
        self.used_mem = 0.0
        self.running_tasks = []

    @property
    def free_cpu(self):
        return self.capacity_cpu - self.used_cpu

    @property
    def free_mem(self):
        return self.capacity_mem - self.used_mem

    def can_host(self, task):
        return (task.requested_cpu <= self.free_cpu) and (task.requested_ram <= self.free_mem)

    def add_task(self, task, current_time):
        self.used_cpu += task.requested_cpu
        self.used_mem += task.requested_ram
        task.start_time = current_time
        task.finish_time = current_time + task.estimated_runtime
        self.running_tasks.append(task)

    def update(self, current_time):
        finished_tasks = 0
        still_running = []
        for t in self.running_tasks:
            if t.finish_time <= current_time:
                self.used_cpu -= t.requested_cpu
                self.used_mem -= t.requested_ram
                finished_tasks += 1
            else:
                still_running.append(t)
        self.running_tasks = still_running
        return finished_tasks
