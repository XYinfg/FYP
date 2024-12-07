# metrics.py
import numpy as np

def compute_job_completion_times(tasks):
    completion_times = [t.finish_time - t.arrival_time for t in tasks if t.finish_time is not None]
    if completion_times:
        avg = np.mean(completion_times)
        median = np.median(completion_times)
        return avg, median
    return None, None

def compute_resource_utilization(machines):
    total_cpu = sum(m.capacity_cpu for m in machines)
    used_cpu = sum(m.used_cpu for m in machines)
    cpu_util = used_cpu / total_cpu if total_cpu > 0 else 0.0
    return cpu_util

def compute_fairness(tasks):
    completion_times = [t.finish_time - t.arrival_time for t in tasks if t.finish_time]
    if len(completion_times) == 0:
        return None
    completion_times = np.array(completion_times)
    numerator = (completion_times.sum())**2
    denominator = len(completion_times)* (completion_times**2).sum()
    return numerator/denominator

def compute_cost(tasks, machines):
    # Simple cost model: sum of (requested_cpu * runtime * factor)
    total_cost = 0.0
    cpu_cost_factor = 0.01
    for t in tasks:
        if t.finish_time and t.start_time:
            runtime = t.finish_time - t.start_time
            total_cost += t.requested_cpu * runtime * cpu_cost_factor
    return total_cost

def summarize_results(simulation):
    tasks = simulation.tasks
    machines = simulation.machines
    avg_ct, median_ct = compute_job_completion_times(tasks)
    fairness = compute_fairness(tasks)
    utilization = compute_resource_utilization(machines)
    cost = compute_cost(tasks, machines)

    return {
        "avg_completion_time": avg_ct,
        "median_completion_time": median_ct,
        "fairness": fairness,
        "utilization": utilization,
        "cost": cost
    }
