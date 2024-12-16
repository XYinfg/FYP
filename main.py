from src.data_loader import DataLoader
from src.models.task import Task
from src.models.machine import Machine
from src.evaluation import evaluate_schedulers
from src.utils import normalize_timestamps
from src.config import (
    TASK_EVENTS_DIR, 
    MACHINE_EVENTS_DIR, 
    MAX_TASK_FILES,
    LOGGING_LEVEL,
    LOGGING_FORMAT,
    LOGGING_DATEFMT
)
import logging

def main():
    # Configure logging
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATEFMT
    )
    logger = logging.getLogger("Main")
    logger.info("Simulation started.")
    
    # Load data
    data_loader = DataLoader(TASK_EVENTS_DIR, MACHINE_EVENTS_DIR)
    logger.debug("Loading machine events.")
    machine_df = data_loader.load_machine_events()
    logger.debug("Machine events loaded successfully.")
    
    logger.debug("Loading task events.")
    task_df = data_loader.load_task_events(max_files=MAX_TASK_FILES)
    logger.debug("Task events loaded successfully.")
    
    # Construct machine objects from ADD events
    add_events = machine_df[machine_df['event_type'] == 0]
    machines = []
    logger.debug(f"Processing {len(add_events)} ADD machine events.")
    for _, row in add_events.iterrows():
        # Assign instance_type_index based on some logic; default to 0 if unknown
        instance_type_index = 0  # Defaulting to first instance type
        # If ADD events contain platform_id or other info to determine instance type, implement logic here
        m = Machine(
            machine_id=row['machine_id'], 
            cpu_capacity=row['cpu_capacity'], 
            mem_capacity=row['mem_capacity'], 
            platform_id=row['platform_id'],
            instance_type_index=instance_type_index  # Assigning instance type index
        )
        machines.append(m)
    logger.info(f"Initialized {len(machines)} machines.")
    
    # Construct tasks from SUBMIT events (event_type == 0)
    submit_events = task_df[task_df['event_type'] == 0]
    tasks = []
    logger.debug(f"Processing {len(submit_events)} SUBMIT task events.")
    for _, row in submit_events.iterrows():
        t = Task(
            job_id=row['job_id'],
            task_index=row['task_index'],
            submit_time=row['timestamp'],
            cpu_req=row['cpu_req'],
            mem_req=row['mem_req'],
            disk_req=row['disk_req'],
            priority=row['priority']
        )
        tasks.append(t)
    logger.info(f"Initialized {len(tasks)} tasks.")
    
    # Normalize submit_time to start at 0
    logger.debug("Normalizing task submit times.")
    normalize_timestamps(tasks)
    logger.debug("Task submit times normalized.")
    
    # Evaluate schedulers
    logger.info("Starting scheduler evaluations.")
    results = evaluate_schedulers(tasks, machines)
    logger.info("Scheduler evaluations completed.")
    
    # Print results
    logger.info("Comparison Results:")
    for sched, res in results.items():
        logger.info(f"{sched}: {res}")
    
    logger.info("Simulation finished.")

if __name__ == "__main__":
    main()