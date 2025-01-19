import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles preprocessing of Google Cluster Data, including:
    - Calculating task runtimes
    - Processing machine states
    - Cleaning and validating data
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor"""
        self.start_events = {'SUBMIT', 'SCHEDULE'}
        self.end_events = {'FINISH', 'FAIL', 'KILL', 'LOST'}

    def calculate_task_runtimes(self, task_events: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate runtime for each task based on start and end events.
        
        Args:
            task_events: DataFrame containing task events
            
        Returns:
            DataFrame with task runtimes and other relevant information
        """
        # Sort by timestamp to ensure correct event ordering
        task_events = task_events.sort_values('timestamp')
        
        # Group by job_id and task_index
        task_groups = task_events.groupby(['job_id', 'task_index'])
        
        runtime_records = []
        invalid_tasks = 0
        
        for (job_id, task_index), group in task_groups:
            # Find valid start events (SUBMIT or SCHEDULE)
            start_events = group[group['event_type'].isin(self.start_events)]
            if start_events.empty:
                invalid_tasks += 1
                continue
                
            # Find valid end events (FINISH, FAIL, KILL, or LOST)
            end_events = group[group['event_type'].isin(self.end_events)]
            if end_events.empty:
                invalid_tasks += 1
                continue
            
            # Get the first valid start and end events
            start_event = start_events.iloc[0]
            end_event = end_events.iloc[0]
            
            # Validate timestamps
            if start_event['timestamp'] >= end_event['timestamp']:
                invalid_tasks += 1
                continue
            
            runtime = (end_event['timestamp'] - start_event['timestamp']).total_seconds()
            
            # Only include tasks with positive runtime
            if runtime > 0:
                runtime_records.append({
                    'job_id': job_id,
                    'task_index': task_index,
                    'start_time': start_event['timestamp'],
                    'end_time': end_event['timestamp'],
                    'runtime': runtime,
                    'end_event_type': end_event['event_type'],
                    'cpu_request': start_event['cpu_request'],
                    'memory_request': start_event['memory_request'],
                    'scheduling_class': start_event['scheduling_class'],
                    'priority': start_event['priority']
                })
        
        if invalid_tasks > 0:
            logger.warning(f"Found {invalid_tasks} tasks with invalid timestamps or missing events")
        
        result_df = pd.DataFrame(runtime_records)
        
        # Log runtime statistics
        if not result_df.empty:
            logger.info(f"Runtime statistics:")
            logger.info(f"  Mean runtime: {result_df['runtime'].mean():.2f} seconds")
            logger.info(f"  Median runtime: {result_df['runtime'].median():.2f} seconds")
            logger.info(f"  Min runtime: {result_df['runtime'].min():.2f} seconds")
            logger.info(f"  Max runtime: {result_df['runtime'].max():.2f} seconds")
            logger.info(f"  Total valid tasks: {len(result_df)}")
        
        return result_df

    def process_machine_states(self, machine_events: pd.DataFrame) -> pd.DataFrame:
        """
        Process machine events to create a timeline of machine states.
        
        Args:
            machine_events: DataFrame containing machine events
            
        Returns:
            DataFrame with machine state information
        """
        # Sort by timestamp and machine_id
        machine_events = machine_events.sort_values(['timestamp', 'machine_id'])
        
        # Initialize state tracking
        machine_states = []
        current_states = {}
        
        for _, event in machine_events.iterrows():
            machine_id = event['machine_id']
            event_type = event['event_type']
            
            if event_type == 'ADD':
                current_states[machine_id] = {
                    'machine_id': machine_id,
                    'start_time': event['timestamp'],
                    'platform_id': event['platform_id'],
                    'cpu_capacity': event['cpu_capacity'],
                    'memory_capacity': event['memory_capacity'],
                    'state': 'ACTIVE'
                }
            
            elif event_type == 'UPDATE' and machine_id in current_states:
                # Record the previous state
                prev_state = current_states[machine_id].copy()
                prev_state['end_time'] = event['timestamp']
                machine_states.append(prev_state)
                
                # Update the state
                current_states[machine_id].update({
                    'start_time': event['timestamp'],
                    'cpu_capacity': event['cpu_capacity'],
                    'memory_capacity': event['memory_capacity']
                })
            
            elif event_type == 'REMOVE' and machine_id in current_states:
                # Record the final state for this machine
                final_state = current_states[machine_id].copy()
                final_state['end_time'] = event['timestamp']
                final_state['state'] = 'REMOVED'
                machine_states.append(final_state)
                
                # Remove the machine from current states
                del current_states[machine_id]
        
        # Add any remaining active states
        end_time = machine_events['timestamp'].max()
        for machine_id, state in current_states.items():
            final_state = state.copy()
            final_state['end_time'] = end_time
            machine_states.append(final_state)
        
        return pd.DataFrame(machine_states)

    def create_runtime_bins(self, runtimes: pd.Series, n_bins: int = 10) -> pd.DataFrame:
        """
        Create runtime bins using exponential binning as per Stratus algorithm.
        
        Args:
            runtimes: Series containing task runtimes
            n_bins: Number of bins to create
            
        Returns:
            DataFrame with bin boundaries and task counts
        """
        if runtimes.empty:
            return pd.DataFrame(columns=['bin_start', 'bin_end', 'task_count', 'bin_label'])
            
        # Calculate exponential bin edges
        min_runtime = max(1, runtimes.min())  # Ensure minimum is at least 1 second
        max_runtime = runtimes.max()
        
        bin_edges = np.exp(
            np.linspace(
                np.log(min_runtime),
                np.log(max_runtime),
                n_bins + 1
            )
        )
        
        # Create bins and count tasks in each bin
        task_counts, edges = np.histogram(runtimes, bins=bin_edges)
        
        # Create DataFrame with bin information
        bins_df = pd.DataFrame({
            'bin_start': edges[:-1],
            'bin_end': edges[1:],
            'task_count': task_counts
        })
        
        # Add bin labels
        bins_df['bin_label'] = [
            f'Bin {i}: [{start:.1f}s, {end:.1f}s)'
            for i, (start, end) in enumerate(zip(bins_df['bin_start'], bins_df['bin_end']))
        ]
        
        return bins_df

    def validate_data(self, task_runtimes: pd.DataFrame, machine_states: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate preprocessed data for consistency and completeness.
        
        Args:
            task_runtimes: DataFrame containing task runtime information
            machine_states: DataFrame containing machine state information
            
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        validation_messages = []
        is_valid = True
        
        # Validate task runtimes
        if task_runtimes.empty:
            validation_messages.append("No valid task runtimes found")
            is_valid = False
        else:
            # Check for negative runtimes
            neg_runtimes = task_runtimes[task_runtimes['runtime'] <= 0]
            if not neg_runtimes.empty:
                validation_messages.append(f"Found {len(neg_runtimes)} tasks with non-positive runtimes")
                is_valid = False
            
            # Check for missing resource requests
            missing_resources = task_runtimes[
                (task_runtimes['cpu_request'].isna()) |
                (task_runtimes['memory_request'].isna())
            ]
            if not missing_resources.empty:
                validation_messages.append(f"Found {len(missing_resources)} tasks with missing resource requests")
                is_valid = False
        
        # Validate machine states
        if machine_states.empty:
            validation_messages.append("No valid machine states found")
            is_valid = False
        else:
            # Check for overlapping machine states
            for machine_id in machine_states['machine_id'].unique():
                machine_timeline = machine_states[machine_states['machine_id'] == machine_id].sort_values('start_time')
                
                for i in range(len(machine_timeline) - 1):
                    if machine_timeline.iloc[i]['end_time'] > machine_timeline.iloc[i + 1]['start_time']:
                        validation_messages.append(f"Found overlapping states for machine {machine_id}")
                        is_valid = False
                        break
        
        return is_valid, validation_messages

    def preprocess_data(self, 
                       task_events: pd.DataFrame, 
                       machine_events: pd.DataFrame,
                       n_runtime_bins: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Preprocess both task events and machine events data.
        
        Args:
            task_events: DataFrame containing task events
            machine_events: DataFrame containing machine events
            n_runtime_bins: Number of runtime bins to create
            
        Returns:
            Dictionary containing preprocessed DataFrames
        """
        # Handle empty input data
        if task_events.empty and machine_events.empty:
            empty_runtime_df = pd.DataFrame(columns=[
                'job_id', 'task_index', 'start_time', 'end_time', 'runtime',
                'end_event_type', 'cpu_request', 'memory_request',
                'scheduling_class', 'priority'
            ])
            empty_machine_df = pd.DataFrame(columns=[
                'machine_id', 'start_time', 'end_time', 'platform_id',
                'cpu_capacity', 'memory_capacity', 'state'
            ])
            empty_bins_df = pd.DataFrame(columns=[
                'bin_start', 'bin_end', 'task_count', 'bin_label'
            ])
            return {
                'task_runtimes': empty_runtime_df,
                'machine_states': empty_machine_df,
                'runtime_bins': empty_bins_df
            }

        logger.info("Calculating task runtimes...")
        task_runtimes = self.calculate_task_runtimes(task_events)
        
        logger.info("Processing machine states...")
        machine_states = self.process_machine_states(machine_events)
        
        logger.info("Creating runtime bins...")
        runtime_bins = self.create_runtime_bins(
            task_runtimes['runtime'] if not task_runtimes.empty else pd.Series([]),
            n_bins=n_runtime_bins
        )
        
        logger.info("Validating preprocessed data...")
        is_valid, messages = self.validate_data(task_runtimes, machine_states)
        
        if not is_valid:
            logger.warning("Data validation issues found:")
            for message in messages:
                logger.warning(message)
        
        return {
            'task_runtimes': task_runtimes,
            'machine_states': machine_states,
            'runtime_bins': runtime_bins
        }