#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified HFT Backtest Runner

This script demonstrates how to use the high-performance backtest framework
for HFT strategies across multiple time periods (1 month, 6 months, 1 year, 5 years).
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simplified_hft_backtest.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define test periods
TEST_PERIODS = {
    "1_month": {
        "start_date": "2024-02-01",
        "end_date": "2024-03-01",
        "description": "Recent market conditions"
    },
    "6_months": {
        "start_date": "2023-09-01",
        "end_date": "2024-03-01",
        "description": "Medium-term market conditions"
    },
    "1_year": {
        "start_date": "2023-03-01",
        "end_date": "2024-03-01",
        "description": "Full market cycle"
    },
    "5_years": {
        "start_date": "2019-03-01",
        "end_date": "2024-03-01",
        "description": "Long-term market conditions including COVID volatility"
    }
}

# Define components to test
COMPONENTS = ["markov_chain", "ml_assessment", "rnn_strategy", "full_ensemble"]

def create_framework():
    """Create the backtest framework with appropriate settings"""
    try:
        # Add parent directory to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import the framework
        from backtest.advanced_component_backtest import AdvancedBacktestFramework
        
        # Initialize framework
        framework = AdvancedBacktestFramework(
            symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            timeframes=["1m", "5m", "15m", "1h", "4h"],
            initial_balance=10000.0,
            max_position_size=100.0,
            test_periods=TEST_PERIODS
        )
        
        logger.info("Framework initialized successfully")
        return framework
    except Exception as e:
        logger.error(f"Error creating framework: {e}")
        return None

def step1_run_period_backtest(framework, period_name, output_dir):
    """Step 1: Run backtest for a specific period"""
    logger.info(f"Step 1: Running backtest for period: {period_name}")
    
    # Create period-specific output directory
    period_dir = os.path.join(output_dir, period_name)
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    try:
        # Run high-performance backtest for this period
        results = framework.run_high_performance_backtest(
            period_name=period_name,
            component_names=COMPONENTS,
            output_dir=period_dir
        )
        
        # Save period-specific results
        results_path = os.path.join(period_dir, f"{period_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Completed backtest for period: {period_name}")
        logger.info(f"Results saved to: {results_path}")
        
        return results
    except Exception as e:
        logger.error(f"Error running backtest for period {period_name}: {e}")
        return None

def step2_analyze_hft_metrics(framework, period_name, component_name, output_dir):
    """Step 2: Analyze HFT-specific metrics"""
    logger.info(f"Step 2: Analyzing HFT metrics for {component_name} in {period_name}")
    
    # Create period-specific output directory
    period_dir = os.path.join(output_dir, period_name)
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    try:
        # Analyze HFT performance
        hft_results = framework.analyze_hft_performance(
            period_name=period_name,
            component_name=component_name,
            min_trades_per_day=400,  # 100k+ executions per year
            latency_ms=5,            # 5ms latency simulation
            slippage_pips=0.5        # 0.5 pips slippage simulation
        )
        
        # Save HFT analysis results
        hft_path = os.path.join(period_dir, f"{component_name}_hft_metrics.json")
        with open(hft_path, 'w') as f:
            json.dump(hft_results, f, indent=4)
        
        logger.info(f"HFT metrics saved to: {hft_path}")
        return hft_results
    except Exception as e:
        logger.error(f"Error analyzing HFT metrics for {component_name} in {period_name}: {e}")
        return None

def step3_run_monte_carlo(framework, period_name, component_name, output_dir):
    """Step 3: Run Monte Carlo simulations for market microstructure variations"""
    logger.info(f"Step 3: Running Monte Carlo simulations for {component_name} in {period_name}")
    
    # Create period-specific output directory
    period_dir = os.path.join(output_dir, period_name)
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    try:
        # Run Monte Carlo simulations
        mc_results = framework.run_monte_carlo_simulation(
            period_name=period_name,
            component_name=component_name,
            num_simulations=1000,
            parameter_variation=0.1,  # 10% parameter variation
            simulate_microstructure=True  # Simulate market microstructure variations
        )
        
        # Save Monte Carlo results
        mc_path = os.path.join(period_dir, f"{component_name}_monte_carlo.json")
        with open(mc_path, 'w') as f:
            json.dump(mc_results, f, indent=4)
        
        logger.info(f"Monte Carlo results saved to: {mc_path}")
        return mc_results
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulations for {component_name} in {period_name}: {e}")
        return None

def step4_run_cross_validation(framework, period_name, component_name, output_dir):
    """Step 4: Run cross-validation with walk-forward analysis"""
    logger.info(f"Step 4: Running cross-validation for {component_name} in {period_name}")
    
    # Create period-specific output directory
    period_dir = os.path.join(output_dir, period_name)
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    try:
        # Run cross-validation
        cv_results = framework.run_cross_validation(
            period_name=period_name,
            component_name=component_name,
            folds=5,
            walk_forward=True  # Use walk-forward analysis
        )
        
        # Save cross-validation results
        cv_path = os.path.join(period_dir, f"{component_name}_cross_validation.json")
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        logger.info(f"Cross-validation results saved to: {cv_path}")
        return cv_results
    except Exception as e:
        logger.error(f"Error running cross-validation for {component_name} in {period_name}: {e}")
        return None

def step5_analyze_performance_metrics(framework, period_name, component_name, output_dir):
    """Step 5: Analyze advanced performance metrics"""
    logger.info(f"Step 5: Analyzing performance metrics for {component_name} in {period_name}")
    
    # Create period-specific output directory
    period_dir = os.path.join(output_dir, period_name)
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    try:
        # Analyze performance metrics
        metrics_results = framework.analyze_advanced_metrics(
            period_name=period_name,
            component_name=component_name,
            calculate_confidence_intervals=True,
            analyze_equity_curve=True,
            analyze_drawdowns=True,
            test_statistical_significance=True
        )
        
        # Save performance metrics results
        metrics_path = os.path.join(period_dir, f"{component_name}_advanced_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_results, f, indent=4)
        
        logger.info(f"Advanced metrics saved to: {metrics_path}")
        return metrics_results
    except Exception as e:
        logger.error(f"Error analyzing performance metrics for {component_name} in {period_name}: {e}")
        return None

def run_complete_workflow(framework, periods, output_dir):
    """Run the complete HFT backtest workflow for all periods"""
    logger.info("Starting complete HFT backtest workflow")
    
    # Create main output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Store all results
    all_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "periods": periods,
        "components": COMPONENTS,
        "results": {}
    }
    
    # Run workflow for each period
    for period_name in periods:
        logger.info(f"Processing period: {period_name}")
        period_results = {}
        
        # Step 1: Run period backtest
        backtest_results = step1_run_period_backtest(framework, period_name, output_dir)
        period_results["backtest"] = backtest_results
        
        # For the full ensemble component, run additional analyses
        component_name = "full_ensemble"
        
        # Step 2: Analyze HFT metrics
        hft_results = step2_analyze_hft_metrics(framework, period_name, component_name, output_dir)
        period_results["hft_metrics"] = hft_results
        
        # Step 3: Run Monte Carlo simulations
        mc_results = step3_run_monte_carlo(framework, period_name, component_name, output_dir)
        period_results["monte_carlo"] = mc_results
        
        # Step 4: Run cross-validation
        cv_results = step4_run_cross_validation(framework, period_name, component_name, output_dir)
        period_results["cross_validation"] = cv_results
        
        # Step 5: Analyze performance metrics
        metrics_results = step5_analyze_performance_metrics(framework, period_name, component_name, output_dir)
        period_results["performance_metrics"] = metrics_results
        
        # Store period results
        all_results["results"][period_name] = period_results
    
    # Save overall results
    overall_path = os.path.join(output_dir, "overall_workflow_results.json")
    with open(overall_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"Complete workflow results saved to: {overall_path}")
    return all_results

def main():
    """Main function to run the simplified HFT backtest"""
    logger.info("Starting simplified HFT backtest")
    
    # Step 0: Create framework
    framework = create_framework()
    if not framework:
        logger.error("Failed to create framework. Exiting.")
        return
    
    # Define output directory
    output_dir = os.path.join(os.getcwd(), "hft_results")
    
    # Define periods to test
    periods = ["1_month", "6_months", "1_year", "5_years"]
    
    # Run complete workflow
    results = run_complete_workflow(framework, periods, output_dir)
    
    # Print summary
    print("\n===== HFT BACKTEST SUMMARY =====")
    print(f"Tested periods: {', '.join(periods)}")
    print(f"Components: {', '.join(COMPONENTS)}")
    print(f"Results saved to: {output_dir}")
    
    print("\nNext steps:")
    print("1. Review the results in each period directory")
    print("2. Compare performance across different time periods")
    print("3. Implement the best parameters in your trading strategy")
    print("4. Consider running forward tests with the optimized parameters")

if __name__ == "__main__":
    main()
