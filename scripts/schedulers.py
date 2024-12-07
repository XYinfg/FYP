import math
import random
from entities import Machine

class BaseScheduler:
    def schedule(self, pending_tasks, machines, current_time):
        raise NotImplementedError

class FCFS(BaseScheduler):
    def schedule(self, pending_tasks, machines, current_time):
        assignments = {}
        for task in pending_tasks:
            for m in machines:
                if m.can_host(task):
                    assignments[task.task_id] = m.machine_id
                    break
        return assignments

class MinMin(BaseScheduler):
    def schedule(self, pending_tasks, machines, current_time):
        assignments = {}
        for task in pending_tasks:
            best_machine = None
            best_finish_time = float('inf')
            for m in machines:
                if m.can_host(task):
                    finish_time = current_time + task.estimated_runtime
                    if finish_time < best_finish_time:
                        best_finish_time = finish_time
                        best_machine = m
            if best_machine:
                assignments[task.task_id] = best_machine.machine_id
        return assignments

class StratusScheduler(BaseScheduler):
    """
    Stratus Algorithm (Simplified)
    Key Features Implemented:
    - Runtime binning of tasks
    - Two-phase packing (up and down packing)
    - Dynamic scaling (basic heuristic to add machines)
    - Cost-awareness (simplified)
    - VM clearing (heuristic)
    """
    def __init__(self, instance_types=None, bin_base=2, max_bin=5, cost_cpu_factor=0.01, cost_mem_factor=0.001):
        self.bin_base = bin_base
        self.max_bin = max_bin
        self.cost_cpu_factor = cost_cpu_factor
        self.cost_mem_factor = cost_mem_factor
        if instance_types is None:
            self.instance_types = [
                {"type": "small",  "cpu": 1.0, "mem": 1.0, "cost_per_hour": 0.02},
                {"type": "medium", "cpu": 4.0, "mem": 8.0, "cost_per_hour": 0.08},
                {"type": "large",  "cpu": 16.0,"mem": 64.0,"cost_per_hour": 0.32}
            ]
        else:
            self.instance_types = instance_types

    def schedule(self, pending_tasks, machines, current_time):
        # Bin tasks by runtime
        bin_map = {}
        for t in pending_tasks:
            b = self.get_bin_for_task(t)
            bin_map.setdefault(b, []).append(t)

        assignments = {}
        machine_bins = self.classify_machines_by_bin(machines)

        for b, tasks_in_bin in bin_map.items():
            for task in tasks_in_bin:
                assigned = False
                # Up-packing
                for bin_candidate in range(b, self.max_bin+1):
                    if bin_candidate in machine_bins:
                        for m in machine_bins[bin_candidate]:
                            if m.can_host(task):
                                assignments[task.task_id] = m.machine_id
                                assigned = True
                                break
                    if assigned:
                        break

                # Down-packing
                if not assigned:
                    for bin_candidate in range(b-1, -1, -1):
                        if bin_candidate in machine_bins:
                            for m in machine_bins[bin_candidate]:
                                if m.can_host(task):
                                    assignments[task.task_id] = m.machine_id
                                    assigned = True
                                    break
                        if assigned:
                            break

                # Scaling if not assigned
                if not assigned:
                    new_vm = self.acquire_vm_for_task(task)
                    machines.append(new_vm)
                    machine_bins = self.classify_machines_by_bin(machines)
                    assignments[task.task_id] = new_vm.machine_id

        # VM clearing (simple heuristic)
        self.clear_underutilized_vms(machines, current_time)

        return assignments

    def get_bin_for_task(self, task):
        runtime = task.estimated_runtime
        if runtime < 1:
            return 0
        b = int(math.log(runtime, self.bin_base))
        if b > self.max_bin:
            b = self.max_bin
        return b

    def classify_machines_by_bin(self, machines):
        machine_bins = {}
        for m in machines:
            if m.running_tasks:
                max_bin = max(self.get_bin_for_task(t) for t in m.running_tasks)
            else:
                max_bin = 0
            machine_bins.setdefault(max_bin, []).append(m)
        return machine_bins

    def acquire_vm_for_task(self, task):
        suitable_types = [it for it in self.instance_types if it["cpu"] >= task.requested_cpu and it["mem"] >= task.requested_ram]
        if not suitable_types:
            chosen = max(self.instance_types, key=lambda x: x["cpu"])
        else:
            chosen = min(suitable_types, key=lambda x: x["cost_per_hour"])

        return Machine(
            machine_id=f"vm_{chosen['type']}_{random.randint(1000,9999)}",
            capacity_cpu=chosen["cpu"],
            capacity_mem=chosen["mem"],
            cost_per_hour=chosen["cost_per_hour"]
        )

    def clear_underutilized_vms(self, machines, current_time):
        underutilized = [m for m in machines if (m.free_cpu/m.capacity_cpu > 0.9) and len(m.running_tasks) > 0]
        for m in underutilized:
            tasks_to_move = m.running_tasks[:]
            for t in tasks_to_move:
                candidate = None
                for other in machines:
                    if other is not m and other.can_host(t):
                        candidate = other
                        break
                if candidate:
                    m.running_tasks.remove(t)
                    m.used_cpu -= t.requested_cpu
                    m.used_mem -= t.requested_ram
                    t.start_time = current_time
                    t.finish_time = current_time + t.estimated_runtime
                    candidate.add_task(t, current_time)
