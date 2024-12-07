import pandas as pd
import os

TASK_EVENTS_DIR = "../data/task_events"
MACHINE_EVENTS_DIR = "../data/machine_events"
OUTPUT_DIR = "../data/processed"
TIME_START = 0 
TIME_END = 10**3
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_task_events(limit=None):
    # Load and concatenate all task_event files
    all_files = [os.path.join(TASK_EVENTS_DIR, f) for f in os.listdir(TASK_EVENTS_DIR) if f.endswith(".csv.gz")]
    df_list = []
    for f in all_files:
        df = pd.read_csv(f, compression='gzip', header=None)
        df.columns = ["timestamp", "missing_info", "job_id", "task_index", "machine_id", "event_type", 
                      "user_name", "scheduling_class", "priority", "requested_cpu", "requested_ram", 
                      "requested_disk", "different_machine_constraint"]
        df_list.append(df)
    tasks = pd.concat(df_list, ignore_index=True)
    if limit:
        tasks = tasks.head(limit)
    return tasks

def load_machine_events():
    all_files = [os.path.join(MACHINE_EVENTS_DIR, f) for f in os.listdir(MACHINE_EVENTS_DIR) if f.endswith(".csv.gz")]
    df_list = []
    for f in all_files:
        df = pd.read_csv(f, compression='gzip', header=None)
        df.columns = ["timestamp", "machine_id", "event_type", "platform_id", "capacity_cpu", "capacity_mem"]
        df_list.append(df)
    machines = pd.concat(df_list, ignore_index=True)
    return machines

if __name__ == "__main__":
    tasks = load_task_events(limit=10000)
    machines = load_machine_events()

    tasks = tasks[(tasks["timestamp"] >= TIME_START) & (tasks["timestamp"] <= TIME_END)].copy()
    machines = machines[(machines["timestamp"] >= TIME_START) & (machines["timestamp"] <= TIME_END)].copy()

    # Save processed subsets for simulation
    tasks.to_csv(os.path.join(OUTPUT_DIR, "tasks_processed.csv"), index=False)
    machines.to_csv(os.path.join(OUTPUT_DIR, "machines_processed.csv"), index=False)
