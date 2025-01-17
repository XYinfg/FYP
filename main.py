from pathlib import Path
import logging
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.scheduler.stratus_scheduler import StratusScheduler, SchedulerConfig
from src.models.scaler import InstanceType
from src.scheduler.baseline_schedulers import FCFSScheduler, BestFitScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_schedulers():
    """Set up all schedulers with configuration"""
    # Define instance types (based on typical cloud offerings)
    instance_types = [
        InstanceType("small", 2.0, 4.0, 0.1),     # 2 vCPU, 4GB, $0.10/hr
        InstanceType("medium", 4.0, 8.0, 0.2),    # 4 vCPU, 8GB, $0.20/hr
        InstanceType("large", 8.0, 16.0, 0.4),    # 8 vCPU, 16GB, $0.40/hr
        InstanceType("xlarge", 16.0, 32.0, 0.8)   # 16 vCPU, 32GB, $0.80/hr
    ]
    
    # Initialize scheduler configuration
    config = SchedulerConfig(
        scheduling_interval=10.0,
        min_instances=1,
        max_instances=100,
        runtime_bins=10,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        cooldown_period=300
    )
    
    return {
        'Stratus': StratusScheduler(config, instance_types),
        'FCFS': FCFSScheduler(config, instance_types),
        'BestFit': BestFitScheduler(config, instance_types)
    }

def create_output_directory():
    """Create output directory for results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("results") / f'sim_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_metrics(metrics_df: pd.DataFrame, output_dir: Path, scheduler_name: str):
    """Create visualizations of scheduler metrics"""
    # 1. Resource utilization over time
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['timestamp'], metrics_df['total_machines'], label='Machines')
    plt.plot(metrics_df['timestamp'], metrics_df['active_tasks'], label='Active Tasks')
    plt.plot(metrics_df['timestamp'], metrics_df['pending_tasks'], label='Pending Tasks')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title(f'Resource Utilization Over Time - {scheduler_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'resource_utilization_{scheduler_name.lower()}.png')
    plt.close()
    
    # 2. Cost analysis
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['timestamp'], metrics_df['estimated_hourly_cost'], label='Hourly Cost')
    plt.xlabel('Time')
    plt.ylabel('Cost ($)')
    plt.title(f'Cost Analysis Over Time - {scheduler_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'cost_analysis_{scheduler_name.lower()}.png')
    plt.close()

def main():
    # Setup
    data_dir = Path("data")
    output_dir = create_output_directory()
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = DataLoader(data_dir)
        preprocessor = DataPreprocessor()
        schedulers = setup_schedulers()
        
        # Load and preprocess data
        logger.info("Loading data...")
        sample_data = data_loader.load_sample_data(n_tasks=10)  # Start with small sample
        
        logger.info("Preprocessing data...")
        processed_data = preprocessor.preprocess_data(
            sample_data['task_events'],
            sample_data['machine_events']
        )
        
        # Process with each scheduler
        all_metrics = {}
        for name, scheduler in schedulers.items():
            logger.info(f"\nTesting {name} scheduler...")
            
            if name == 'Stratus':
                # Train the runtime estimator for Stratus
                logger.info("Training runtime estimator...")
                training_data = processed_data['task_runtimes']
                scheduler.runtime_estimator.train(training_data)
                
                # Initialize runtime bins
                logger.info("Initializing runtime bins...")
                scheduler.runtime_bins.create_bins(training_data['runtime'])
            
            # Start scheduler
            logger.info(f"Starting {name} scheduler...")
            if hasattr(scheduler, 'start'):
                scheduler.start()
            
            # Add initial machines
            logger.info("Adding initial machines...")
            for i in range(3):  # Start with 3 machines
                if name == 'Stratus':
                    scheduler.add_machine(scheduler.scaler.instance_types["small"])
                else:
                    scheduler.add_machine()
            
            # Submit tasks
            logger.info("Submitting tasks...")
            task_runtimes = processed_data['task_runtimes']
            for _, task in task_runtimes.iterrows():
                scheduler.submit_task(
                    job_id=str(task['job_id']),
                    cpu_request=task['cpu_request'],
                    memory_request=task['memory_request'],
                    priority=1
                )
                time.sleep(0.1)
            
            # Run for a while to allow processing
            logger.info("Processing tasks...")
            simulation_time = 300  # 5 minutes
            metrics = []
            
            start_time = time.time()
            while time.time() - start_time < simulation_time:
                # Process tasks for baseline schedulers
                if name != 'Stratus':
                    scheduler.schedule_tasks()
                
                # Collect metrics
                metrics.append({
                    'timestamp': datetime.now(),
                    'total_machines': len(scheduler.resource_manager.machines),
                    'active_tasks': len(scheduler.resource_manager.running_tasks),
                    'pending_tasks': len(scheduler.resource_manager.pending_tasks),
                    'completed_tasks': len(scheduler.resource_manager.completed_tasks),
                    'failed_tasks': len(scheduler.resource_manager.failed_tasks),
                    'estimated_hourly_cost': (
                        scheduler.get_cost_analysis()['hourly_cost'] 
                        if hasattr(scheduler, 'get_cost_analysis') 
                        else 0.0
                    )
                })
                
                time.sleep(10)
            
            # Stop scheduler
            logger.info(f"Stopping {name} scheduler...")
            if hasattr(scheduler, 'stop'):
                scheduler.stop()
            
            # Save metrics
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(output_dir / f'metrics_{name.lower()}.csv', index=False)
            plot_metrics(metrics_df, output_dir, name)
            all_metrics[name] = metrics_df
            
            # Save final statistics
            final_stats = {
                'total_tasks_processed': len(task_runtimes),
                'completed_tasks': len(scheduler.resource_manager.completed_tasks),
                'failed_tasks': len(scheduler.resource_manager.failed_tasks),
                'total_cost': (
                    scheduler.get_cost_analysis()['hourly_cost'] * (simulation_time / 3600)
                    if hasattr(scheduler, 'get_cost_analysis')
                    else 0.0
                )
            }
            pd.Series(final_stats).to_csv(output_dir / f'final_stats_{name.lower()}.csv')
        
        # Create comparison plots
        logger.info("Creating comparison plots...")
        plot_scheduler_comparison(all_metrics, output_dir)
        
        logger.info(f"Simulation completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise

def plot_scheduler_comparison(all_metrics: dict, output_dir: Path):
    """Create comparison plots for all schedulers"""
    # Prepare comparison data
    comparison_data = []
    for scheduler_name, metrics_df in all_metrics.items():
        final_metrics = metrics_df.iloc[-1]
        comparison_data.append({
            'Scheduler': scheduler_name,
            'Completed Tasks': final_metrics['completed_tasks'],
            'Active Machines': final_metrics['total_machines'],
            'Hourly Cost': final_metrics['estimated_hourly_cost']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    metrics_to_plot = [
        ('Completed Tasks', 'Number of Tasks'),
        ('Active Machines', 'Number of Machines'),
        ('Hourly Cost', 'Cost ($)')
    ]
    
    for metric, ylabel in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=comparison_df, x='Scheduler', y=metric)
        plt.title(f'{metric} by Scheduler')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'comparison_{metric.lower().replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    main()