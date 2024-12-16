# src/utils.py

import logging

def convert_timestamp(ts, start_ts=None):
    """
    Converts timestamp from microseconds to seconds, 
    and optionally normalizes relative to a start timestamp.
    """
    if start_ts is None:
        return ts / 1_000_000.0
    else:
        return (ts - start_ts) / 1_000_000.0

def normalize_timestamps(task_list):
    """
    Given a list of tasks, normalize their submit_time so that 
    the earliest submit time is zero.
    """
    logger = logging.getLogger("Utils")
    if not task_list:
        logger.debug("No tasks to normalize.")
        return
    min_ts = min(t.submit_time for t in task_list)
    logger.debug(f"Normalizing timestamps with min_ts: {min_ts}.")
    for t in task_list:
        t.submit_time = (t.submit_time - min_ts) / 1_000_000.0
    logger.debug("Task timestamps normalized.")

def get_task_runtime_mapping(finish_events_df):
    """
    Placeholder function for mapping task runtimes from finish events.
    """
    logger = logging.getLogger("Utils")
    logger.debug("Generating task runtime mapping from finish events.")
    runtime_map = {}
    # Implement actual logic if needed
    return runtime_map

def log(message):
    # Simple logging function, if needed
    print(f"[LOG] {message}")
