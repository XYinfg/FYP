import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RuntimeBin:
    """Represents a runtime bin with its properties"""
    index: int
    start: float
    end: float
    label: str
    task_count: int = 0
    total_cpu: float = 0.0
    total_memory: float = 0.0

class RuntimeBinsManager:
    """
    Manages runtime bins for the Stratus scheduler.
    Provides consistent binning across different components.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize the RuntimeBinsManager.
        
        Args:
            n_bins: Number of runtime bins to create
        """
        self.n_bins = n_bins
        self.bins: List[RuntimeBin] = []
        self.bin_edges: np.ndarray = None
    
    def create_bins(self, runtimes: pd.Series) -> None:
        """
        Create runtime bins based on actual runtime distribution.
        
        Args:
            runtimes: Series containing task runtimes
        """
        if runtimes.empty:
            logger.warning("No runtimes provided for bin creation")
            return
        
        # Calculate exponential bin edges
        min_runtime = max(1, runtimes.min())
        min_runtime = runtimes.min()
        max_runtime = runtimes.max()
        
        self.bin_edges = np.exp(
            np.linspace(
                np.log(min_runtime),
                np.log(max_runtime),
                self.n_bins + 1
            )
        )
        
        # Create bins
        self.bins = [
            RuntimeBin(
                index=i,
                start=self.bin_edges[i],
                end=self.bin_edges[i + 1],
                label=f"Bin {i}: [{self.bin_edges[i]:.1f}s, {self.bin_edges[i + 1]:.1f}s)"
            )
            for i in range(self.n_bins)
        ]
    
    def get_bin_index(self, runtime: float) -> int:
        """
        Get the bin index for a given runtime.
        
        Args:
            runtime: Task runtime in seconds
            
        Returns:
            Bin index
        """
        if self.bin_edges is None:
            raise ValueError("Bins not initialized. Call create_bins first.")
        
        return min(
            np.digitize(runtime, self.bin_edges) - 1,
            self.n_bins - 1
        )
    
    def add_task_to_bin(self, 
                       runtime: float, 
                       cpu_request: float, 
                       memory_request: float) -> None:
        """
        Add a task to its appropriate bin.
        
        Args:
            runtime: Task runtime
            cpu_request: Task CPU request
            memory_request: Task memory request
        """
        bin_idx = self.get_bin_index(runtime)
        runtime_bin = self.bins[bin_idx]
        
        runtime_bin.task_count += 1
        runtime_bin.total_cpu += cpu_request
        runtime_bin.total_memory += memory_request
    
    def get_bin_stats(self) -> pd.DataFrame:
        """
        Get statistics for all bins.
        
        Returns:
            DataFrame containing bin statistics
        """
        stats = []
        for bin_ in self.bins:
            stats.append({
                'bin_index': bin_.index,
                'start_time': bin_.start,
                'end_time': bin_.end,
                'label': bin_.label,
                'task_count': bin_.task_count,
                'total_cpu': bin_.total_cpu,
                'total_memory': bin_.total_memory,
                'avg_cpu_per_task': bin_.total_cpu / bin_.task_count if bin_.task_count > 0 else 0,
                'avg_memory_per_task': bin_.total_memory / bin_.task_count if bin_.task_count > 0 else 0
            })
        
        return pd.DataFrame(stats)
    
    def get_bin_edges(self) -> List[float]:
        """
        Get the bin edge values.
        
        Returns:
            List of bin edge values
        """
        return list(self.bin_edges) if self.bin_edges is not None else []
    
    def get_bin_labels(self) -> List[str]:
        """
        Get the bin labels.
        
        Returns:
            List of bin labels
        """
        return [bin_.label for bin_ in self.bins]
    
    def analyze_bin_distribution(self) -> Dict[str, float]:
        """
        Analyze the distribution of tasks across bins.
        
        Returns:
            Dictionary containing distribution metrics
        """
        if not self.bins:
            return {}
        
        total_tasks = sum(bin_.task_count for bin_ in self.bins)
        if total_tasks == 0:
            return {
                'entropy': 0.0,
                'most_common_bin': None,
                'empty_bins': self.n_bins,
                'max_bin_utilization': 0.0
            }
        
        # Calculate distribution entropy
        probs = [bin_.task_count / total_tasks for bin_ in self.bins]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        
        # Find most common bin
        most_common = max(self.bins, key=lambda x: x.task_count)
        
        return {
            'entropy': entropy,
            'most_common_bin': most_common.index,
            'empty_bins': sum(1 for bin_ in self.bins if bin_.task_count == 0),
            'max_bin_utilization': max(probs)
        }
    
    def clear_bins(self) -> None:
        """Reset all bin statistics"""
        for bin_ in self.bins:
            bin_.task_count = 0
            bin_.total_cpu = 0.0
            bin_.total_memory = 0.0