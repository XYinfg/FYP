# run_experiment.py
import pandas as pd
import pickle, os
from entities import Task, Machine
from schedulers import FCFS, MinMin, StratusScheduler
from simulation import Simulation

os.makedirs("./results/outputs", exist_ok=True)

tasks_df = pd.read_csv("./data/processed/tasks_processed.csv")
machines_df = pd.read_csv("./data/processed/machines_processed.csv")

initial_machines = machines_df[machines_df["event_type"] == 0]

# Build tasks list once:
full_tasks = []
for _, row in tasks_df.iterrows():
    if row["event_type"] == 0:  # Submitted tasks
        t = Task(job_id=row["job_id"], task_index=row["task_index"], arrival_time=row["timestamp"],
                 requested_cpu=row["requested_cpu"], requested_ram=row["requested_ram"], runtime=100)
        full_tasks.append(t)

# Function to re-initialize tasks and machines to ensure each simulation is fresh.
def get_fresh_data():
    machines = [Machine(machine_id=row["machine_id"], capacity_cpu=row["capacity_cpu"], capacity_mem=row["capacity_mem"])
                for _, row in initial_machines.iterrows()]

    # Create a fresh copy of tasks
    tasks_copy = [Task(t.job_id, t.task_index, t.arrival_time, t.requested_cpu, t.requested_ram, t.estimated_runtime)
                  for t in full_tasks]
    return tasks_copy, machines

# Run Stratus Simulation
tasks_stratus, machines_stratus = get_fresh_data()
print(f"Running Stratus Simulation: {len(tasks_stratus)} tasks, {len(machines_stratus)} machines")
stratus_sim = Simulation(tasks_stratus, machines_stratus, StratusScheduler(), debug=True)
stratus_events = stratus_sim.run(end_time=10**9)
with open("./results/outputs/stratus_sim.pkl", "wb") as f:
    pickle.dump(stratus_sim, f)
print("Stratus Simulation Completed.")

# Run FCFS Simulation
tasks_fcfs, machines_fcfs = get_fresh_data()
print(f"Running FCFS Simulation: {len(tasks_fcfs)} tasks, {len(machines_fcfs)} machines")
fcfs_sim = Simulation(tasks_fcfs, machines_fcfs, FCFS(), debug=True)
fcfs_events = fcfs_sim.run(end_time=10**9)
with open("./results/outputs/fcfs_sim.pkl", "wb") as f:
    pickle.dump(fcfs_sim, f)
print("FCFS Simulation Completed.")

# Run MinMin Simulation
tasks_minmin, machines_minmin = get_fresh_data()
print(f"Running MinMin Simulation: {len(tasks_minmin)} tasks, {len(machines_minmin)} machines")
minmin_sim = Simulation(tasks_minmin, machines_minmin, MinMin(), debug=True)
minmin_events = minmin_sim.run(end_time=10**9)
with open("./results/outputs/minmin_sim.pkl", "wb") as f:
    pickle.dump(minmin_sim, f)
print("MinMin Simulation Completed.")
