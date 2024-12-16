# src/data_loader.py

import os
import pandas as pd
import logging

class DataLoader:
    def __init__(self, task_events_dir, machine_events_dir):
        self.task_events_dir = task_events_dir
        self.machine_events_dir = machine_events_dir
        self.logger = logging.getLogger("DataLoader")
    
    def load_machine_events(self):
        self.logger.debug("Starting to load machine events.")
        all_files = [
            os.path.join(self.machine_events_dir, f) 
            for f in os.listdir(self.machine_events_dir) 
            if f.endswith('.csv.gz')
        ]
        self.logger.debug(f"Found {len(all_files)} machine event files.")
        df_list = []
        for file in all_files:
            self.logger.debug(f"Loading machine event file: {file}")
            df = pd.read_csv(file, header=None, compression='gzip')
            # Assign column names according to specification:
            # machine_events: timestamp, machine_id, event_type, platform_id, cpu_capacity, mem_capacity
            df.columns = ['timestamp', 'machine_id', 'event_type', 'platform_id', 'cpu_capacity', 'mem_capacity']
            df_list.append(df)
        machine_df = pd.concat(df_list).sort_values(by='timestamp')
        self.logger.debug("All machine events loaded and concatenated.")
        return machine_df
    
    def load_task_events(self, max_files=5):
        self.logger.debug("Starting to load task events.")
        all_files = [
            os.path.join(self.task_events_dir, f) 
            for f in os.listdir(self.task_events_dir) 
            if f.endswith('.csv.gz')
        ]
        self.logger.debug(f"Found {len(all_files)} task event files.")
        df_list = []
        for i, file in enumerate(sorted(all_files)):
            if i >= max_files:
                self.logger.debug(f"Reached max_files limit: {max_files}. Stopping further loading.")
                break
            self.logger.debug(f"Loading task event file: {file}")
            df = pd.read_csv(file, header=None, compression='gzip')
            # Assign column names according to specification:
            # task_events: timestamp, missing_info, job_id, task_index, machine_id, event_type, user, sched_class, priority, cpu_req, mem_req, disk_req, diff_machine_constraint
            df.columns = [
                'timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id',
                'event_type', 'user', 'sched_class', 'priority', 'cpu_req', 'mem_req',
                'disk_req', 'diff_machine_constraint'
            ]
            df_list.append(df)
        task_df = pd.concat(df_list).sort_values(by='timestamp')
        self.logger.debug("All task events loaded and concatenated.")
        return task_df
