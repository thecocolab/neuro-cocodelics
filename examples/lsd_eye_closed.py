#!/usr/bin/env python3
"""
Example script for running ML analysis on LSD closed eyes data.

This script demonstrates how to run the machine learning pipeline
for analyzing LSD closed eyes data using the configuration file.
"""

import subprocess
import sys
import os
import logging
import warnings
import argparse

# Configure environment for Compute Canada multiprocessing before importing anything
def setup_compute_canada_environment():
    """Configure environment variables for optimal performance on Compute Canada."""
    
    # Detect available resources
    total_cpus = os.cpu_count() or 1
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        total_cpus = min(total_cpus, int(slurm_cpus))
    
    # Set conservative resource limits
    safe_cpus = max(1, min(total_cpus // 2, 8))  # Use half available cores, max 8
    
    # Configure environment variables
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(safe_cpus))
    os.environ.setdefault('OMP_NUM_THREADS', '1')  # Avoid thread conflicts
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
    os.environ.setdefault('JOBLIB_MULTIPROCESSING', '1')
    
    # Set multiprocessing start method
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # start method already set
    
    # Suppress multiprocessing warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
    
    print(f"Configured for Compute Canada: Using {safe_cpus} cores (detected {total_cpus} total)")
    return safe_cpus

# Setup environment before any heavy imports
safe_cores = setup_compute_canada_environment()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lsd_closed_eyes_analysis(config_path=None):
    """
    Run the ML analysis for LSD closed eyes data using the baseline config.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the config file. If None, uses the default baseline config.
    """
    # Use provided config path or default
    if config_path is None:
        config_path = "machine-learning/configs/lsd_closed_baseline.yml"
    
    # Path to the ML runner script (relative to project root)
    ml_script_path = "machine-learning/run_ml.py"
    
    # Check if files exist
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    if not os.path.exists(ml_script_path):
        logger.error(f"ML script not found: {ml_script_path}")
        return False
    
    # Run the ML analysis
    try:
        logger.info(f"Starting LSD closed eyes ML analysis with config: {config_path}")
        cmd = [sys.executable, ml_script_path, "--config", config_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("Analysis completed successfully!")
        logger.info("Output:")
        print(result.stdout)
        
        if result.stderr:
            logger.warning("Warnings/Errors:")
            print(result.stderr)
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis failed with return code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """
    Main function to run the LSD closed eyes analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LSD closed eyes ML analysis")
    parser.add_argument("--config", "-c", 
                       help="Path to config YAML file", 
                       default="machine-learning/configs/lsd_closed_baseline.yml")
    args = parser.parse_args()
    
    logger.info("LSD Closed Eyes ML Analysis Example")
    logger.info("=" * 40)
    logger.info(f"Using config: {args.config}")
    
    success = run_lsd_closed_eyes_analysis(args.config)
    
    if success:
        logger.info("Analysis completed successfully!")
        logger.info("Results should be saved in: ./results/lsd_closed/")
    else:
        logger.error("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
