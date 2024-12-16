# src/config.py

import logging

# Paths
TASK_EVENTS_DIR = "data/task_events"
MACHINE_EVENTS_DIR = "data/machine_events"

# Simulation parameters
MAX_TASK_FILES = 2  # Limit number of task event files to load for testing
TIME_UNIT = 1_000_000.0  # Convert microseconds to seconds (1e6)
DEFAULT_ESTIMATED_RUNTIME = 10.0

# Stratus-related configurations
RUNTIME_BINS = [1, 2, 4, 8, 16, 32, 64]  # In seconds, can adjust as needed
INSTANCE_TYPES = [
    (64, 128, 1.0),  # (cpu_capacity, mem_capacity, price)
    (32, 64, 0.5),
    (16, 32, 0.25)
]

LOW_UTILIZATION_THRESHOLD = 0.5
UTILIZATION_CHECK_INTERVAL = 300  # in seconds
SCALE_IN_COOLDOWN = 600  # in seconds

# Logging configuration
LOGGING_LEVEL = logging.DEBUG
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGING_DATEFMT = '%Y-%m-%d %H:%M:%S'
