from collections import deque
from tqdm import tqdm

class Simulation:
    def __init__(self, tasks, machines, scheduler, debug=False):
        self.tasks = tasks
        self.machines = machines
        self.scheduler = scheduler
        self.current_time = 0
        self.tasks.sort(key=lambda x: x.arrival_time)
        self.pending_tasks = deque(self.tasks)
        self.running = True
        self.total_tasks = len(tasks)
        self.debug = debug

    def run(self, end_time=None):
        events = []
        tasks_finished = 0

        if self.debug:
            print(f"[DEBUG] Starting simulation with {self.total_tasks} tasks and {len(self.machines)} machines.")
            if self.total_tasks == 0:
                print("[DEBUG] No tasks to process. Simulation will end immediately.")
            if len(self.machines) == 0:
                print("[DEBUG] No machines available. Tasks cannot be scheduled.")

        pbar = tqdm(total=self.total_tasks, desc="Simulating", unit="task")

        iteration = 0
        while self.running:
            iteration += 1
            if end_time and self.current_time > end_time:
                print("[DEBUG] Reached end_time without completing all tasks.")
                break

            # Check if all tasks completed
            if not self.pending_tasks and all(len(m.running_tasks) == 0 for m in self.machines):
                # All tasks done
                if self.debug:
                    print("[DEBUG] All tasks have completed.")
                break

            finished_this_round = 0
            for m in self.machines:
                finished = m.update(self.current_time)
                finished_this_round += finished
                if self.debug and iteration % 1000 == 0:
                    print(f"[DEBUG] Time={self.current_time}, Machine={m.machine_id}, finished={finished}")

            if finished_this_round > 0:
                tasks_finished += finished_this_round
                pbar.update(finished_this_round)

            if self.debug and iteration % 1000 == 0:
                print(f"[DEBUG] Iteration={iteration}, Time={self.current_time}, Finished={tasks_finished}/{self.total_tasks}, Pending={len(self.pending_tasks)}")

            # Get tasks that have arrived
            ready_tasks = []
            while self.pending_tasks and self.pending_tasks[0].arrival_time <= self.current_time:
                ready_tasks.append(self.pending_tasks.popleft())

            assigned_count = 0
            if ready_tasks:
                if self.debug:
                    print(f"[DEBUG] {len(ready_tasks)} tasks ready at time {self.current_time}. Attempting to schedule.")
                assignments = self.scheduler.schedule(ready_tasks, self.machines, self.current_time)
                for t in ready_tasks:
                    if t.task_id in assignments:
                        machine_id = assignments[t.task_id]
                        machine = next((m for m in self.machines if m.machine_id == machine_id), None)
                        if machine:
                            machine.add_task(t, self.current_time)
                            events.append((self.current_time, t.task_id, machine_id))
                            assigned_count += 1
                        else:
                            print("[DEBUG] Assigned machine not found for task", t.task_id)
                    else:
                        # Could not schedule now, do not re-queue
                        if self.debug:
                            print(f"[DEBUG] Task {t.task_id} could not be scheduled at time {self.current_time}.")

            if assigned_count == 0 and finished_this_round == 0 and not ready_tasks:
                # No progress made this round. Possibly stuck.
                print("[DEBUG] No progress made this cycle. Possibly stuck. Current time:", self.current_time)
                print("[DEBUG] Pending tasks:", len(self.pending_tasks))
                print("[DEBUG] Running tasks:", sum(len(m.running_tasks) for m in self.machines))
                print("[DEBUG] Breaking out of simulation to avoid infinite loop.")
                break

            # Advance time
            next_finish_times = [t.finish_time for m in self.machines for t in m.running_tasks]
            next_arrival_time = self.pending_tasks[0].arrival_time if self.pending_tasks else None
            candidates = []
            if next_finish_times:
                candidates.append(min(next_finish_times))
            if next_arrival_time is not None:
                candidates.append(next_arrival_time)

            if candidates:
                next_time = min(t for t in candidates if t >= self.current_time)
                if next_time == self.current_time and assigned_count == 0 and finished_this_round == 0:
                    # Time is not advancing
                    print("[DEBUG] Time cannot advance and no progress was made. Ending simulation.")
                    break
                self.current_time = next_time
            else:
                # No future events
                break

        pbar.close()
        return events
