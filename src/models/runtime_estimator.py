import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuntimeEstimator:
    """
    Handles task runtime estimation for the Stratus scheduler.
    Implements runtime prediction and adjustment mechanisms.
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the RuntimeEstimator.
        
        Args:
            model_dir: Directory to save/load trained models
        """
        self.model_dir = model_dir if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_request', 'memory_request', 'priority'
        ]

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: DataFrame containing task information
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        features = []
        feature_names = []
        
        # Basic task features
        for col in self.feature_columns:
            if col in df.columns:
                features.append(df[col].values.reshape(-1, 1))
                feature_names.append(col)
        
        # Combine all features
        X = np.hstack(features)
        
        return X, feature_names

    def train(self, task_data: pd.DataFrame) -> None:
        """
        Train the runtime estimation model using historical task data.
        
        Args:
            task_data: DataFrame containing task information and actual runtimes
        """
        logger.info("Preparing training data...")
        X, feature_names = self.prepare_features(task_data)
        y = task_data['runtime'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info("Training random forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        if self.model_dir:
            logger.info("Saving model and scaler...")
            joblib.dump(self.model, self.model_dir / 'runtime_model.joblib')
            joblib.dump(self.scaler, self.model_dir / 'feature_scaler.joblib')
            
            # Save feature names for reference
            with open(self.model_dir / 'feature_names.txt', 'w') as f:
                f.write('\n'.join(feature_names))

    def load_model(self) -> bool:
        """
        Load a previously trained model and scaler.
        
        Returns:
            bool: True if model was loaded successfully
        """
        try:
            model_path = self.model_dir / 'runtime_model.joblib'
            scaler_path = self.model_dir / 'feature_scaler.joblib'
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, task_data: pd.DataFrame) -> np.ndarray:
        """
        Predict runtimes for new tasks.
        
        Args:
            task_data: DataFrame containing task information
            
        Returns:
            Array of predicted runtimes
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("No trained model available")
        
        X, _ = self.prepare_features(task_data)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 1.0)
        
        return predictions

    def adjust_runtime_estimate(self, 
                              initial_estimate: float,
                              elapsed_time: float,
                              completion_ratio: Optional[float] = None) -> float:
        """
        Adjust runtime estimate based on actual execution progress.
        
        Args:
            initial_estimate: Initial runtime estimate
            elapsed_time: Time spent executing so far
            completion_ratio: Optional progress indicator (0-1)
            
        Returns:
            Adjusted runtime estimate
        """
        # If we've already exceeded the initial estimate
        if elapsed_time >= initial_estimate:
            # Double the elapsed time as new estimate
            return elapsed_time * 2.0
        
        if completion_ratio is not None and completion_ratio > 0:
            # Use completion ratio to estimate total runtime
            estimated_total = elapsed_time / completion_ratio
            # Take max of initial estimate and completion-based estimate
            return max(initial_estimate, estimated_total)
        
        # Conservative adjustment: assume at least half the initial estimate remains
        remaining_estimate = max(
            initial_estimate - elapsed_time,
            initial_estimate / 2.0
        )
        return elapsed_time + remaining_estimate

    def get_runtime_bin(self, runtime: float, bin_edges: np.ndarray) -> int:
        """
        Determine which runtime bin a task belongs to.
        
        Args:
            runtime: Task runtime
            bin_edges: Array of bin boundaries
            
        Returns:
            Bin index
        """
        return np.digitize(runtime, bin_edges) - 1

    def analyze_estimation_accuracy(self, 
                                  actual_runtimes: np.ndarray,
                                  predicted_runtimes: np.ndarray) -> Dict[str, float]:
        """
        Analyze the accuracy of runtime estimates.
        
        Args:
            actual_runtimes: Array of actual task runtimes
            predicted_runtimes: Array of predicted runtimes
            
        Returns:
            Dictionary containing various error metrics
        """
        abs_errors = np.abs(actual_runtimes - predicted_runtimes)
        rel_errors = abs_errors / actual_runtimes
        
        return {
            'mean_absolute_error': np.mean(abs_errors),
            'median_absolute_error': np.median(abs_errors),
            'mean_relative_error': np.mean(rel_errors),
            'median_relative_error': np.median(rel_errors),
            'rmse': np.sqrt(np.mean(np.square(abs_errors))),
            'overestimation_rate': np.mean(predicted_runtimes > actual_runtimes)
        }

    def update_estimates(self, 
                        task_states: pd.DataFrame,
                        current_time: pd.Timestamp) -> pd.DataFrame:
        """
        Update runtime estimates for running tasks based on current progress.
        
        Args:
            task_states: DataFrame containing current task states
            current_time: Current timestamp
            
        Returns:
            DataFrame with updated runtime estimates
        """
        updated_states = task_states.copy()
        
        # Only update running tasks
        running_mask = task_states['state'] == 'RUNNING'
        if not running_mask.any():
            return updated_states
        
        running_tasks = task_states[running_mask]
        
        for idx, task in running_tasks.iterrows():
            elapsed_time = (current_time - task['start_time']).total_seconds()
            
            # Get completion ratio if available
            completion_ratio = task.get('completion_ratio', None)
            
            # Adjust runtime estimate
            new_estimate = self.adjust_runtime_estimate(
                task['estimated_runtime'],
                elapsed_time,
                completion_ratio
            )
            
            updated_states.loc[idx, 'estimated_runtime'] = new_estimate
            updated_states.loc[idx, 'remaining_time'] = new_estimate - elapsed_time
        
        return updated_states