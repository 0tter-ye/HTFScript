#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HFT High-Performance Backtest Runner

This script implements a step-by-step workflow for high-frequency trading backtesting
across multiple time periods (1 month, 6 months, 1 year, 5 years).
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the framework
from backtest.advanced_component_backtest import AdvancedBacktestFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hft_backtest.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_period_backtest(framework, config, period_name):
    """Run backtest for a specific period"""
    logger.info(f"Starting backtest for period: {period_name}")
    
    # Create period-specific output directory
    output_dir = os.path.join(config.get("output_dir", "hft_results"), period_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run high-performance backtest for this period
    results = framework.run_high_performance_backtest(
        period_name=period_name,
        component_names=config.get("components"),
        output_dir=output_dir
    )
    
    # Save period-specific results
    results_path = os.path.join(output_dir, f"{period_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Completed backtest for period: {period_name}")
    logger.info(f"Results saved to: {results_path}")
    
    return results

def calculate_confidence_intervals(framework, config, period_name, component_name):
    """Calculate statistical confidence intervals for performance metrics"""
    logger.info(f"Calculating confidence intervals for {component_name} in {period_name}")
    
    # Get period info
    period_info = config["test_periods"].get(period_name)
    if not period_info:
        logger.error(f"Invalid period: {period_name}")
        return {}
    
    # Run Monte Carlo simulations
    mc_results = framework.run_monte_carlo_simulation(
        period_name=period_name,
        component_name=component_name,
        num_simulations=config.get("monte_carlo", {}).get("simulations", 1000),
        parameter_variation=config.get("monte_carlo", {}).get("parameter_variation_percent", 10) / 100
    )
    
    # Calculate confidence intervals
    confidence_level = config.get("monte_carlo", {}).get("confidence_interval", 0.95)
    alpha = 1 - confidence_level
    
    # Extract profit factors and win rates from simulations
    pf_values = [sim.get("profit_factor", 0) for sim in mc_results.get("simulations", [])]
    wr_values = [sim.get("win_rate", 0) for sim in mc_results.get("simulations", [])]
    
    # Calculate confidence intervals
    pf_mean = np.mean(pf_values)
    pf_std = np.std(pf_values)
    pf_lower, pf_upper = stats.norm.interval(confidence_level, loc=pf_mean, scale=pf_std)
    
    wr_mean = np.mean(wr_values)
    wr_std = np.std(wr_values)
    wr_lower, wr_upper = stats.norm.interval(confidence_level, loc=wr_mean, scale=wr_std)
    
    # Create confidence interval results
    ci_results = {
        "profit_factor": {
            "mean": pf_mean,
            "std_dev": pf_std,
            "lower_bound": max(0, pf_lower),
            "upper_bound": pf_upper,
            "confidence_level": confidence_level
        },
        "win_rate": {
            "mean": wr_mean,
            "std_dev": wr_std,
            "lower_bound": max(0, wr_lower),
            "upper_bound": min(100, wr_upper),
            "confidence_level": confidence_level
        }
    }
    
    # Save confidence interval results
    output_dir = os.path.join(config.get("output_dir", "hft_results"), period_name)
    ci_path = os.path.join(output_dir, f"{component_name}_confidence_intervals.json")
    with open(ci_path, 'w') as f:
        json.dump(ci_results, f, indent=4)
    
    logger.info(f"Confidence intervals saved to: {ci_path}")
    return ci_results

def analyze_hft_metrics(framework, config, period_name, component_name):
    """Analyze HFT-specific metrics"""
    logger.info(f"Analyzing HFT metrics for {component_name} in {period_name}")
    
    # Get period info
    period_info = config["test_periods"].get(period_name)
    if not period_info:
        logger.error(f"Invalid period: {period_name}")
        return {}
    
    # Run HFT analysis
    hft_results = framework.analyze_hft_performance(
        period_name=period_name,
        component_name=component_name,
        min_trades_per_day=config.get("optimization", {}).get("min_trades_per_day", 400),
        latency_ms=config.get("optimization", {}).get("latency_simulation_ms", 5),
        slippage_pips=config.get("optimization", {}).get("slippage_simulation_pips", 0.5)
    )
    
    # Save HFT analysis results
    output_dir = os.path.join(config.get("output_dir", "hft_results"), period_name)
    hft_path = os.path.join(output_dir, f"{component_name}_hft_metrics.json")
    with open(hft_path, 'w') as f:
        json.dump(hft_results, f, indent=4)
    
    logger.info(f"HFT metrics saved to: {hft_path}")
    return hft_results

def compare_periods(config, periods):
    """Compare results across different time periods"""
    logger.info("Comparing results across time periods")
    
    # Initialize comparison data
    comparison = {
        "periods": periods,
        "components": {},
        "best_component_by_period": {},
        "most_consistent_component": None,
        "parameter_stability": {}
    }
    
    # Load results for each period
    period_results = {}
    for period in periods:
        result_path = os.path.join(config.get("output_dir", "hft_results"), period, f"{period}_results.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                period_results[period] = json.load(f)
    
    # Compare components across periods
    components = config.get("components", [])
    for component in components:
        comparison["components"][component] = {
            "profit_factor": {},
            "win_rate": {},
            "success_rate": {},
            "consistency_score": 0
        }
        
        # Track parameter values across periods
        param_values = {}
        
        # Collect metrics for each period
        success_count = 0
        for period in periods:
            if period in period_results:
                # Get component results
                component_result = period_results[period].get("components", {}).get(component, {})
                
                # Get best optimization method
                best_method = component_result.get("best_optimization")
                if best_method:
                    # Get metrics from best method
                    if best_method == "profit_factor":
                        metrics = component_result.get("profit_factor_optimization", {})
                    elif best_method == "win_rate":
                        metrics = component_result.get("win_rate_optimization", {})
                    elif best_method == "combined_metrics":
                        metrics = component_result.get("combined_metrics_optimization", {})
                    elif best_method == "multi_timeframe":
                        metrics = component_result.get("multi_timeframe_optimization", {})
                    else:
                        metrics = {}
                    
                    # Store metrics
                    comparison["components"][component]["profit_factor"][period] = metrics.get("profit_factor", 0)
                    comparison["components"][component]["win_rate"][period] = metrics.get("win_rate", 0)
                    comparison["components"][component]["success_rate"][period] = 1 if component_result.get("success", False) else 0
                    
                    if component_result.get("success", False):
                        success_count += 1
                    
                    # Track parameters
                    if best_method == "multi_timeframe":
                        best_params = component_result.get("multi_timeframe_optimization", {}).get("best_parameters", {})
                        for param, value in best_params.items():
                            if param not in param_values:
                                param_values[param] = []
                            param_values[param].append(value)
        
        # Calculate consistency score (percentage of periods where component was successful)
        if periods:
            consistency = (success_count / len(periods)) * 100
            comparison["components"][component]["consistency_score"] = consistency
        
        # Calculate parameter stability across periods
        param_stability = {}
        for param, values in param_values.items():
            if len(values) >= 2:
                mean = np.mean(values)
                if mean != 0:
                    cv = np.std(values) / abs(mean)  # Coefficient of variation
                    stability = max(0, 100 * (1 - min(cv, 1)))
                    param_stability[param] = stability
        
        comparison["parameter_stability"][component] = param_stability
    
    # Determine best component for each period
    for period in periods:
        best_component = None
        best_score = 0
        
        for component in components:
            # Calculate score based on profit factor and win rate
            pf = comparison["components"][component]["profit_factor"].get(period, 0)
            wr = comparison["components"][component]["win_rate"].get(period, 0)
            
            # Weight profit factor more heavily
            score = (pf / 50.0) * 0.7 + (wr / 55.0) * 0.3
            
            if score > best_score:
                best_score = score
                best_component = component
        
        comparison["best_component_by_period"][period] = best_component
    
    # Determine most consistent component
    most_consistent = None
    highest_consistency = 0
    
    for component in components:
        consistency = comparison["components"][component]["consistency_score"]
        if consistency > highest_consistency:
            highest_consistency = consistency
            most_consistent = component
    
    comparison["most_consistent_component"] = most_consistent
    
    # Save comparison results
    comparison_path = os.path.join(config.get("output_dir", "hft_results"), "period_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Period comparison saved to: {comparison_path}")
    return comparison

def visualize_period_comparison(config, comparison):
    """Create visualizations for period comparison"""
    logger.info("Creating period comparison visualizations")
    
    # Create visualization directory
    vis_dir = os.path.join(config.get("output_dir", "hft_results"), "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 1. Profit Factor Comparison
    try:
        periods = comparison.get("periods", [])
        components = list(comparison.get("components", {}).keys())
        
        # Create DataFrame for profit factors
        pf_data = {}
        for component in components:
            pf_data[component] = [
                comparison["components"][component]["profit_factor"].get(period, 0)
                for period in periods
            ]
        
        pf_df = pd.DataFrame(pf_data, index=periods)
        
        # Plot profit factors
        plt.figure(figsize=(12, 8))
        pf_df.plot(kind='bar', ax=plt.gca())
        plt.title("Profit Factor Comparison Across Time Periods")
        plt.ylabel("Profit Factor")
        plt.xlabel("Time Period")
        plt.axhline(y=50, color='red', linestyle='--', label='Target (50)')
        plt.legend(title="Components")
        plt.tight_layout()
        
        pf_path = os.path.join(vis_dir, "profit_factor_comparison.png")
        plt.savefig(pf_path)
        plt.close()
        
        # 2. Win Rate Comparison
        # Create DataFrame for win rates
        wr_data = {}
        for component in components:
            wr_data[component] = [
                comparison["components"][component]["win_rate"].get(period, 0)
                for period in periods
            ]
        
        wr_df = pd.DataFrame(wr_data, index=periods)
        
        # Plot win rates
        plt.figure(figsize=(12, 8))
        wr_df.plot(kind='bar', ax=plt.gca())
        plt.title("Win Rate Comparison Across Time Periods")
        plt.ylabel("Win Rate (%)")
        plt.xlabel("Time Period")
        plt.axhline(y=55, color='red', linestyle='--', label='Target (55%)')
        plt.legend(title="Components")
        plt.tight_layout()
        
        wr_path = os.path.join(vis_dir, "win_rate_comparison.png")
        plt.savefig(wr_path)
        plt.close()
        
        # 3. Consistency Scores
        consistency_scores = [
            comparison["components"][component]["consistency_score"]
            for component in components
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(components, consistency_scores, color='purple')
        plt.title("Component Consistency Across Time Periods")
        plt.ylabel("Consistency Score (%)")
        plt.xlabel("Component")
        plt.axhline(y=75, color='red', linestyle='--', label='Target (75%)')
        plt.legend()
        plt.tight_layout()
        
        consistency_path = os.path.join(vis_dir, "consistency_scores.png")
        plt.savefig(consistency_path)
        plt.close()
        
        logger.info(f"Visualizations saved to: {vis_dir}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def main():
    """Main function to run the HFT backtest workflow"""
    # Load configuration
    config_file = "hft_config.json"
    config = load_config(config_file)
    
    # Create output directory
    output_dir = config.get("output_dir", "hft_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize framework
    framework = AdvancedBacktestFramework(
        symbols=config.get("symbols"),
        timeframes=config.get("timeframes"),
        initial_balance=config.get("initial_balance", 10000.0),
        max_position_size=config.get("max_position_size", 100.0),
        test_periods=config.get("test_periods")
    )
    
    # Step 1: Run backtests for each period
    periods = ["1_month", "6_months", "1_year", "5_years"]
    period_results = {}
    
    for period in periods:
        period_results[period] = run_period_backtest(framework, config, period)
    
    # Step 2: Calculate confidence intervals for the full ensemble
    for period in periods:
        calculate_confidence_intervals(framework, config, period, "full_ensemble")
    
    # Step 3: Analyze HFT-specific metrics
    for period in periods:
        analyze_hft_metrics(framework, config, period, "full_ensemble")
    
    # Step 4: Compare results across periods
    comparison = compare_periods(config, periods)
    
    # Step 5: Visualize comparison
    visualize_period_comparison(config, comparison)
    
    # Print summary of results
    print("\n===== HFT BACKTEST SUMMARY =====")
    print(f"Tested periods: {', '.join(periods)}")
    print(f"Most consistent component: {comparison.get('most_consistent_component')}")
    
    print("\nBest component by period:")
    for period, component in comparison.get("best_component_by_period", {}).items():
        print(f"- {period}: {component}")
    
    print("\nResults saved to:", output_dir)
    print("\nNext steps:")
    print("1. Review the visualizations in the 'visualizations' directory")
    print("2. Examine the confidence intervals for statistical significance")
    print("3. Implement the parameters from the most consistent component")
    print("4. Run forward tests with the optimized parameters")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
