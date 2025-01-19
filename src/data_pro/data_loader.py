import pandas as pd
import gzip
from pathlib import Path
from typing import Dict, Optional, Union, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and basic processing of Google Cluster Data.
    """
    
    # Column names for machine events
    MACHINE_COLS = [
        'timestamp', 'machine_id', 'event_type', 
        'platform_id', 'cpu_capacity', 'memory_capacity'
    ]
    
    # Column names for task events
    TASK_COLS = [
        'timestamp', 'missing_info', 'job_id', 'task_index',
        'machine_id', 'event_type', 'user_name', 'scheduling_class',
        'priority', 'cpu_request', 'memory_request', 'disk_request',
        'different_machine_constraint'
    ]
    
    # Event type mappings
    TASK_EVENT_TYPES = {
        0: 'SUBMIT', 1: 'SCHEDULE', 2: 'EVICT',
        3: 'FAIL', 4: 'FINISH', 5: 'KILL',
        6: 'LOST', 7: 'UPDATE_PENDING', 8: 'UPDATE_RUNNING'
    }
    
    MACHINE_EVENT_TYPES = {
        0: 'ADD', 1: 'REMOVE', 2: 'UPDATE'
    }
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the DataLoader with the directory containing the dataset.
        
        Args:
            data_dir: Path to the directory containing machine_events and task_events folders
        """
        self.data_dir = Path(data_dir)
        self.machine_events_dir = self.data_dir / 'machine_events'
        self.task_events_dir = self.data_dir / 'task_events'
        
        # Verify directories exist
        if not self.machine_events_dir.exists():
            raise FileNotFoundError(f"Machine events directory not found: {self.machine_events_dir}")
        if not self.task_events_dir.exists():
            raise FileNotFoundError(f"Task events directory not found: {self.task_events_dir}")

    def read_gzip_csv(self, file_path: Path, columns: List[str]) -> pd.DataFrame:
        """
        Read a gzipped CSV file into a pandas DataFrame.
        
        Args:
            file_path: Path to the gzipped CSV file
            columns: List of column names to use
            
        Returns:
            DataFrame containing the CSV data
        """
        try:
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, names=columns, header=None)
            return df
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def load_machine_events(self, file_name: str = 'part-00000-of-00001.csv.gz') -> pd.DataFrame:
        """
        Load machine events data from the specified file.
        
        Args:
            file_name: Name of the machine events file to load
            
        Returns:
            DataFrame containing machine events data
        """
        file_path = self.machine_events_dir / file_name
        df = self.read_gzip_csv(file_path, self.MACHINE_COLS)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        
        # Map event types to their string representations
        df['event_type'] = df['event_type'].map(self.MACHINE_EVENT_TYPES)
        
        return df

    def validate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean timestamp data.
        
        Args:
            df: DataFrame containing event data
            
        Returns:
            Cleaned DataFrame
        """
        total_rows = len(df)
        zero_timestamps = (df['timestamp'] == 0).sum()
        zero_percentage = (zero_timestamps / total_rows) * 100
        
        logger.info(f"Total events: {total_rows}")
        logger.info(f"Events with timestamp=0: {zero_timestamps} ({zero_percentage:.2f}%)")
        
        # Option 1: Filter out zero timestamps
        df_filtered = df[df['timestamp'] != 0]
        
        # Option 2: Replace zero timestamps with the minimum non-zero timestamp
        min_valid_timestamp = df[df['timestamp'] > 0]['timestamp'].min()
        df_filtered.loc[df_filtered['timestamp'] == 0, 'timestamp'] = min_valid_timestamp
        
        # Convert timestamp to datetime
        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], unit='us')
        
        # Sort by timestamp to ensure chronological order
        df_filtered = df_filtered.sort_values('timestamp')
        
        logger.info(f"Events after cleaning: {len(df_filtered)}")
        return df_filtered

    def load_task_events(self, file_name: str = 'part-00000-of-00500.csv.gz') -> pd.DataFrame:
        """
        Load task events data from the specified file.
        
        Args:
            file_name: Name of the task events file to load
            
        Returns:
            DataFrame containing task events data
        """
        file_path = self.task_events_dir / file_name
        df = self.read_gzip_csv(file_path, self.TASK_COLS)
        
        # Validate and clean timestamps
        df = self.validate_timestamps(df)
        
        # Map event types to their string representations
        df['event_type'] = df['event_type'].map(self.TASK_EVENT_TYPES)
        
        return df

    def load_sample_data(self, n_tasks: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Load a sample of the data for testing and development purposes.
        
        Args:
            n_tasks: Number of task events to load
            
        Returns:
            Dictionary containing machine_events and task_events DataFrames
        """
        # Load machine events
        machine_events = self.load_machine_events()
        
        # Load and sample task events
        task_events = self.load_task_events()
        
        # Get a sample of unique job_ids
        unique_jobs = task_events['job_id'].unique()
        sampled_jobs = pd.Series(unique_jobs).sample(n=min(n_tasks, len(unique_jobs)))
        
        # Filter task events for sampled jobs
        task_events = task_events[task_events['job_id'].isin(sampled_jobs)]
        
        return {
            'machine_events': machine_events,
            'task_events': task_events
        }