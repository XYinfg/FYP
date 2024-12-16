from src.simulation import Simulation
from src.models.scheduler_fcfs import FCFSScheduler
from src.models.scheduler_minmin import MinMinScheduler
from src.models.scheduler_stratus import StratusScheduler
import logging
from src.config import INSTANCE_TYPES
import copy

def evaluate_schedulers(tasks, machines):
    logger = logging.getLogger("Evaluation")
    results = {}
    
    # FCFS
    logger.info("Evaluating FCFS Scheduler.")
    fcfs = FCFSScheduler()
    sim_fcfs = Simulation(copy.deepcopy(tasks), copy.deepcopy(machines), fcfs)  # Use deepcopy to prevent interference
    fcfs_results, fcfs_machines = sim_fcfs.run()
    logger.debug(f"FCFS Results: {fcfs_results}")
    results['fcfs'] = fcfs_results
    
    # Min-Min
    logger.info("Evaluating Min-Min Scheduler.")
    minmin = MinMinScheduler()
    sim_minmin = Simulation(copy.deepcopy(tasks), copy.deepcopy(machines), minmin)
    minmin_results, minmin_machines = sim_minmin.run()
    logger.debug(f"Min-Min Results: {minmin_results}")
    results['minmin'] = minmin_results
    
    # Stratus
    logger.info("Evaluating Stratus Scheduler.")
    stratus = StratusScheduler(INSTANCE_TYPES)
    sim_stratus = Simulation(copy.deepcopy(tasks), copy.deepcopy(machines), stratus)
    stratus_results, stratus_machines = sim_stratus.run()
    logger.debug(f"Stratus Results: {stratus_results}")
    results['stratus'] = stratus_results
    
    return results
