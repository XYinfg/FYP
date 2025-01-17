import pandas as pd
import numpy as np
from typing import Dict, List, Type, Any, Tuple
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time

from src.scheduler.stratus_scheduler import StratusScheduler, SchedulerConfig
from src.models.scaler import InstanceType
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.utils.metrics import SchedulerMetrics
from src.scheduler.baseline_schedulers import FCFSScheduler, BestFitScheduler

logger = logging.getLogger(__name__)

class SchedulerBenchmark:
    """
    Utility for benchmarking different scheduler implementations.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 output_dir: Path,
                 simulation_duration: int = 3600):  # 1 hour
        """
        Initialize the benchmark.
        
        Args:
            data_dir: Directory containing input data
            output_dir: Directory for results
            simulation_duration: Duration of each simulation in seconds
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.simulation_duration = simulation_duration
        
        # Initialize data components
        self.data_loader = DataLoader(self.data_dir)
        self.preprocessor = DataPreprocessor()
        
        # Define instance types
        self.instance_types = [
            InstanceType("small", 2.0, 4.0, 0.1),     # 2 vCPU, 4GB, $0.10/hr
            InstanceType("medium", 4.0, 8.0, 0.2),    # 4 vCPU, 8GB, $0.20/hr
            InstanceType("large", 8.0, 16.0, 0.4),    # 8 vCPU, 16GB, $0.40/hr
            InstanceType("xlarge", 16.0, 32.0, 0.8)   # 16 vCPU, 32GB, $0.80/hr
        ]
        
        # Define scheduler config
        self.config = SchedulerConfig(
            scheduling_interval=10.0,
            min_instances=1,
            max_instances=100,
            runtime_bins=10,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
    
    def prepare_data(self, sample_size: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for benchmarking"""
        logger.info("Loading data samples...")
        data = self.data_loader.load_sample_data(n_tasks=sample_size)
        
        logger.info("Preprocessing data...")
        processed_data = self.preprocessor.preprocess_data(
            data['task_events'],
            data['machine_events']
        )
        
        return data['task_events'], data['machine_events']
    
    def run_simulation(self, 
                      scheduler: Any,
                      task_data: pd.DataFrame) -> SchedulerMetrics:
        """
        Run simulation with a specific scheduler.
        
        Args:
            scheduler: Scheduler instance
            task_data: Task data to process
            
        Returns:
            Metrics from the simulation
        """
        start_time = time.time()
        task_runtimes = self.preprocessor.calculate_task_runtimes(task_data)
        
        # Submit tasks
        for _, task in task_runtimes.iterrows():
            scheduler.submit_task(
                job_id=str(task['job_id']),
                cpu_request=task['cpu_request'],
                memory_request=task['memory_request'],
                priority=1
            )
            
            # Process tasks
            scheduler.schedule_tasks()
            
            # Check simulation time
            if time.time() - start_time >= self.simulation_duration:
                break
            
            time.sleep(0.1)  # Small delay between submissions
        
        return scheduler.metrics
    
    def run_benchmark(self, task_data: pd.DataFrame) -> Dict[str, SchedulerMetrics]:
        """
        Run benchmark comparing different schedulers.
        
        Args:
            task_data: Task data to process
            
        Returns:
            Dictionary mapping scheduler names to their metrics
        """
        results = {}
        
        # Test Stratus scheduler
        logger.info("Testing Stratus scheduler...")
        stratus = StratusScheduler(self.config, self.instance_types)
        results['Stratus'] = self.run_simulation(stratus, task_data)
        
        # Test FCFS scheduler
        logger.info("Testing FCFS scheduler...")
        fcfs = FCFSScheduler(self.config, self.instance_types)
        results['FCFS'] = self.run_simulation(fcfs, task_data)
        
        # Test Best-Fit scheduler
        logger.info("Testing Best-Fit scheduler...")
        best_fit = BestFitScheduler(self.config, self.instance_types)
        results['BestFit'] = self.run_simulation(best_fit, task_data)
        
        return results
    
    def analyze_results(self, results: Dict[str, SchedulerMetrics]) -> None:
        """
        Analyze and visualize benchmark results.
        
        Args:
            results: Dictionary of scheduler metrics
        """
        # Create output directory
        plots_dir = self.output_dir / 'benchmark_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare comparison data
        comparison_data = []
        for scheduler_name, metrics in results.items():
            stats = metrics.get_full_report()
            
            # Flatten metrics for comparison
            flat_metrics = {
                'scheduler': scheduler_name,
                'avg_waiting_time': stats['task_stats']['avg_waiting_time'],
                'completion_rate': stats['task_stats']['completion_rate'],
                'avg_cpu_util': stats['resource_stats']['avg_cpu_util'],
                'avg_memory_util': stats['resource_stats']['avg_memory_util'],
                'total_cost': stats['resource_stats']['total_cost'],
                'sla_violation_rate': stats['sla_metrics']['sla_violation_rate']
            }
            comparison_data.append(flat_metrics)
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison visualizations
        metrics_to_plot = [
            ('avg_waiting_time', 'Average Waiting Time (s)'),
            ('completion_rate', 'Task Completion Rate'),
            ('avg_cpu_util', 'Average CPU Utilization'),
            ('total_cost', 'Total Cost ($)')
        ]
        
        for metric, title in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='scheduler', y=metric)
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / f'{metric}_comparison.png')
            plt.close()
        
        # Save detailed comparison
        df.to_csv(self.output_dir / 'scheduler_comparison.csv', index=False)
        
        # Save individual scheduler metrics
        for scheduler_name, metrics in results.items():
            scheduler_dir = self.output_dir / scheduler_name.lower()
            scheduler_dir.mkdir(parents=True, exist_ok=True)
            metrics.save_metrics(str(scheduler_dir))
        
        logger.info(f"Benchmark results saved to {self.output_dir}")