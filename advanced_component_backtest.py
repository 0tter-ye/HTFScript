#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Component Backtesting Framework
This script provides a comprehensive backtesting framework for the ultra-optimized strategy
with advanced quantitative components, supporting multi-period testing, component-specific
analysis, Monte Carlo simulations, stress testing, and cross-validation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import multiprocessing
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the backtest runner and advanced components
from .run_ultra_optimized_backtest import run_comprehensive_backtest, generate_optimal_parameters
# Check if advanced_quant_components exists in the backtest directory, otherwise import from core
try:
    from .advanced_quant_components import create_advanced_components
except ImportError:
    # Try to import from core directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from advanced_quant_components import create_advanced_components

# Import the ultra_optimized_strategy from core directory
try:
    from ..core.ultra_optimized_strategy import UltraOptimizedStrategy
except ImportError:
    # Try to import directly
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from ultra_optimized_strategy import UltraOptimizedStrategy

class AdvancedBacktestFramework:
    """
    Advanced backtesting framework for ultra-optimized strategy with advanced quantitative components
    """
    
    def __init__(self, symbols=None, timeframes=None, initial_balance=50000):
        """
        Initialize the advanced backtest framework
        
        Args:
            symbols (list): List of symbols to backtest
            timeframes (list): List of timeframes to backtest
            initial_balance (float): Initial balance
        """
        # Default symbols and timeframes if not provided
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT", "BNBUSDT"]
        self.timeframes = timeframes or ["3m", "5m", "15m"]
        self.initial_balance = initial_balance
        
        # Set up test periods
        self.current_date = datetime.now()
        self.test_periods = {
            "1_month": {
                "start_date": (self.current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Recent market conditions (1 month)"
            },
            "6_months": {
                "start_date": (self.current_date - timedelta(days=180)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Medium-term performance (6 months)"
            },
            "1_year": {
                "start_date": (self.current_date - timedelta(days=365)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Full market cycle (1 year)"
            },
            "5_years": {
                "start_date": (self.current_date - timedelta(days=1825)).strftime('%Y-%m-%d'),
                "end_date": self.current_date.strftime('%Y-%m-%d'),
                "description": "Long-term robustness (5 years)"
            },
            # Add specific market condition periods
            "bull_market": {
                "start_date": "2020-10-01",
                "end_date": "2021-04-01",
                "description": "Bull market phase"
            },
            "bear_market": {
                "start_date": "2021-05-01",
                "end_date": "2022-01-01",
                "description": "Bear market phase"
            },
            "sideways_market": {
                "start_date": "2019-01-01",
                "end_date": "2019-06-01",
                "description": "Sideways/consolidation market phase"
            },
            "high_volatility": {
                "start_date": "2020-03-01",
                "end_date": "2020-05-01",
                "description": "High volatility period (COVID crash and recovery)"
            },
            "low_volatility": {
                "start_date": "2019-06-01",
                "end_date": "2019-12-01",
                "description": "Low volatility period"
            }
        }
        
        # Results storage
        self.results = {}
        self.component_results = {}
        self.monte_carlo_results = {}
        self.stress_test_results = {}
        self.cross_validation_results = {}
        
        # Component weights for testing
        self.component_weights = {
            "default": {
                'markov_confidence_weight': 0.30,
                'ml_confidence_weight': 0.25,
                'fractal_confidence_weight': 0.20,
                'price_action_weight': 0.15,
                'volume_confidence_weight': 0.10
            },
            "markov_focused": {
                'markov_confidence_weight': 0.50,
                'ml_confidence_weight': 0.20,
                'fractal_confidence_weight': 0.15,
                'price_action_weight': 0.10,
                'volume_confidence_weight': 0.05
            },
            "ml_focused": {
                'markov_confidence_weight': 0.20,
                'ml_confidence_weight': 0.50,
                'fractal_confidence_weight': 0.15,
                'price_action_weight': 0.10,
                'volume_confidence_weight': 0.05
            },
            "fractal_focused": {
                'markov_confidence_weight': 0.20,
                'ml_confidence_weight': 0.15,
                'fractal_confidence_weight': 0.50,
                'price_action_weight': 0.10,
                'volume_confidence_weight': 0.05
            },
            "balanced": {
                'markov_confidence_weight': 0.20,
                'ml_confidence_weight': 0.20,
                'fractal_confidence_weight': 0.20,
                'price_action_weight': 0.20,
                'volume_confidence_weight': 0.20
            }
        }
        
        logger.info(f"Initialized Advanced Backtest Framework")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Test periods: {len(self.test_periods)} periods from 1 month to 5 years")
    
    def run_multi_period_backtest(self, optimization_rounds=3):
        """
        Run backtests across multiple time periods
        
        Args:
            optimization_rounds (int): Number of optimization rounds to perform
            
        Returns:
            dict: Results for each period
        """
        logger.info("Starting multi-period backtesting")
        
        period_results = {}
        
        for period_name, period_info in self.test_periods.items():
            logger.info(f"Running backtest for period: {period_info['description']}")
            logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
            
            # Run comprehensive backtest for this period
            results, best_config = run_comprehensive_backtest(
                symbols=self.symbols,
                timeframes=self.timeframes,
                initial_balance=self.initial_balance,
                start_date=period_info['start_date'],
                end_date=period_info['end_date'],
                optimization_rounds=optimization_rounds
            )
            
            # Store results
            period_results[period_name] = {
                "results": results,
                "best_config": best_config,
                "period_info": period_info
            }
            
            # Log summary
            self._log_period_summary(period_name, results)
        
        self.results["multi_period"] = period_results
        return period_results
    
    def _log_period_summary(self, period_name, results):
        """
        Log summary of results for a period
        
        Args:
            period_name (str): Name of the period
            results (dict): Results for the period
        """
        logger.info(f"Summary for period: {period_name}")
        
        # Calculate aggregate metrics
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        total_symbols = 0
        
        for symbol, symbol_results in results.items():
            total_symbols += 1
            for timeframe, timeframe_results in symbol_results.items():
                total_profit += timeframe_results.get('net_profit', 0)
                total_trades += timeframe_results.get('total_trades', 0)
                winning_trades += timeframe_results.get('winning_trades', 0)
        
        # Log aggregate metrics
        logger.info(f"Total symbols tested: {total_symbols}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total profit: ${total_profit:.2f}")
        logger.info(f"Overall win rate: {(winning_trades / total_trades * 100) if total_trades > 0 else 0:.2f}%")
        logger.info("-" * 50)

    def run_component_specific_tests(self, period_name="1_year", optimization_rounds=2):
        """
        Run tests for each advanced component individually to measure their impact
        
        Args:
            period_name (str): Which period to use for testing
            optimization_rounds (int): Number of optimization rounds
            
        Returns:
            dict: Results for each component
        """
        logger.info("Starting component-specific testing")
        
        if period_name not in self.test_periods:
            logger.error(f"Invalid period name: {period_name}")
            return None
        
        period_info = self.test_periods[period_name]
        logger.info(f"Using period: {period_info['description']}")
        logger.info(f"Date range: {period_info['start_date']} to {period_info['end_date']}")
        
        # Components to test individually
        components = [
            {
                "name": "baseline",
                "description": "Baseline strategy without advanced components",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False
                }
            },
            {
                "name": "markov_chains",
                "description": "Markov chain models only",
                "config": {
                    'use_markov_chains': True,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'markov_confidence_weight': 1.0
                }
            },
            {
                "name": "ml_models",
                "description": "ML assessment and rescoring only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': True,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'ml_confidence_weight': 1.0
                }
            },
            {
                "name": "fractal_analysis",
                "description": "Fractal dimensions and Hurst exponent only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': True,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'fractal_confidence_weight': 1.0
                }
            },
            {
                "name": "microstructure",
                "description": "Microstructure noise analysis only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': True,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False
                }
            },
            {
                "name": "volume_analysis",
                "description": "Volume analysis only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': True,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': False,
                    'volume_confidence_weight': 1.0
                }
            },
            {
                "name": "rnn_strategy",
                "description": "RNN strategy selection via pseudo-forest only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': True,
                    'use_bid_ask_imbalance': False
                }
            },
            {
                "name": "bid_ask_imbalance",
                "description": "Bid/ask volume imbalance with L2/L3 data only",
                "config": {
                    'use_markov_chains': False,
                    'use_ml_models': False,
                    'use_fractal_analysis': False,
                    'use_microstructure': False,
                    'use_volume_analysis': False,
                    'use_rnn_strategy_selection': False,
                    'use_bid_ask_imbalance': True
                }
            },
            {
                "name": "full_ensemble",
                "description": "All components with default weights",
                "config": {
                    'use_markov_chains': True,
                    'use_ml_models': True,
                    'use_fractal_analysis': True,
                    'use_microstructure': True,
                    'use_volume_analysis': True,
                    'use_rnn_strategy_selection': True,
                    'use_bid_ask_imbalance': True,
                    'markov_confidence_weight': 0.30,
                    'ml_confidence_weight': 0.25,
                    'fractal_confidence_weight': 0.20,
                    'price_action_weight': 0.15,
                    'volume_confidence_weight': 0.10
                }
            }
        ]
        
        component_results = {}
        
        # Test each component configuration
        for component in components:
            logger.info(f"Testing component: {component['name']} - {component['description']}")
            
            # Create a custom parameter set with this component configuration
            custom_params = self._get_base_parameters()
            custom_params.update(component['config'])
            
            # Ensure position size is capped at 100.0 for safety
            custom_params['max_position_size'] = 100.0
            
            # Run backtest with this component configuration
            results = self._run_component_backtest(
                component['name'],
                custom_params,
                period_info['start_date'],
                period_info['end_date'],
                optimization_rounds
            )
            
            # Store results
            component_results[component['name']] = {
                "description": component['description'],
                "config": component['config'],
                "results": results
            }
            
            # Log component summary
            self._log_component_summary(component['name'], component['description'], results)
        
        # Calculate performance improvement for each component compared to baseline
        if 'baseline' in component_results and 'full_ensemble' in component_results:
            logger.info("Calculating component performance contributions")
            
            baseline_results = component_results['baseline']['results']
            full_results = component_results['full_ensemble']['results']
            
            for component_name, component_data in component_results.items():
                if component_name not in ['baseline', 'full_ensemble']:
                    component_results[component_name]['contribution'] = self._calculate_contribution(
                        baseline_results,
                        component_data['results'],
                        full_results
                    )
        
        self.component_results = component_results
        return component_results
    
    def _get_base_parameters(self):
        """
        Get base parameters for testing
        
        Returns:
            dict: Base parameters
        """
        return {
            # Signal generation parameters
            'confidence_threshold': 0.85,
            'force_signals': True,
            'force_signal_threshold': 0.80,
            'min_signals_per_day': 3,
            'max_signals_per_day': 8,
            
            # Position sizing and risk management - capped for safety
            'position_size_multiplier': 3.0,
            'max_position_size': 100.0,  # Safety cap
            'risk_per_trade': 0.01,
            
            # Trade management
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.20,
            'use_trailing_stop': True,
            'trailing_stop_activation': 0.10,
            'trailing_stop_distance': 0.05,
            
            # Advanced components - all disabled by default
            'use_markov_chains': False,
            'use_ml_models': False,
            'use_fractal_analysis': False,
            'use_microstructure': False,
            'use_volume_analysis': False,
            'use_rnn_strategy_selection': False,
            'use_bid_ask_imbalance': False,
            
            # Component weights - will be overridden
            'markov_confidence_weight': 0.0,
            'ml_confidence_weight': 0.0,
            'fractal_confidence_weight': 0.0,
            'price_action_weight': 1.0,  # Default to price action only
            'volume_confidence_weight': 0.0
        }
    
    def _run_component_backtest(self, component_name, params, start_date, end_date, optimization_rounds):
        """
        Run backtest for a specific component configuration
        
        Args:
            component_name (str): Name of the component
            params (dict): Parameters for the backtest
            start_date (str): Start date
            end_date (str): End date
            optimization_rounds (int): Number of optimization rounds
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Running backtest for component: {component_name}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Create a temporary directory for results
        os.makedirs(f"component_tests/{component_name}", exist_ok=True)
        
        # Aggregate results across symbols and timeframes
        aggregate_results = {}
        
        for symbol in self.symbols:
            aggregate_results[symbol] = {}
            
            for timeframe in self.timeframes:
                logger.info(f"Testing {component_name} on {symbol} {timeframe}")
                
                try:
                    # Initialize backtest runner with custom parameters
                    from .ultra_optimized_backtest import UltraOptimizedBacktest
                    backtest = UltraOptimizedBacktest(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        initial_balance=self.initial_balance
                    )
                    
                    # Set date range
                    backtest.strategy.start_date = start_date
                    backtest.strategy.end_date = end_date
                    
                    # Apply component-specific parameters
                    backtest.strategy.params.update(params)
                    
                    # Run backtest
                    results = backtest.run()
                    
                    # Store results
                    aggregate_results[symbol][timeframe] = results
                    
                except Exception as e:
                    logger.error(f"Error testing {component_name} on {symbol} {timeframe}: {str(e)}")
                    aggregate_results[symbol][timeframe] = {"error": str(e)}
        
        return aggregate_results
    
    def _log_component_summary(self, component_name, description, results):
        """
        Log summary of results for a component
        
        Args:
            component_name (str): Name of the component
            description (str): Description of the component
            results (dict): Results for the component
        """
        logger.info(f"Summary for component: {component_name} - {description}")
        
        # Calculate aggregate metrics
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        profit_factor = 0
        max_drawdown = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_profit += timeframe_results.get('net_profit', 0)
                    total_trades += timeframe_results.get('total_trades', 0)
                    winning_trades += timeframe_results.get('winning_trades', 0)
                    
                    # Track maximum drawdown
                    if 'max_drawdown_pct' in timeframe_results:
                        max_drawdown = max(max_drawdown, timeframe_results['max_drawdown_pct'])
                    
                    # Calculate profit factor
                    if 'gross_profit' in timeframe_results and 'gross_loss' in timeframe_results:
                        if timeframe_results['gross_loss'] != 0:
                            symbol_pf = timeframe_results['gross_profit'] / abs(timeframe_results['gross_loss'])
                            profit_factor = max(profit_factor, symbol_pf)
        
        # Log aggregate metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total profit: ${total_profit:.2f}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Max drawdown: {max_drawdown*100:.2f}%")
        logger.info("-" * 50)
    
    def _calculate_contribution(self, baseline_results, component_results, full_results):
        """
        Calculate the performance contribution of a component
        
        Args:
            baseline_results (dict): Baseline results
            component_results (dict): Component results
            full_results (dict): Full ensemble results
            
        Returns:
            dict: Contribution metrics
        """
        # Extract key metrics
        baseline_profit = self._extract_total_profit(baseline_results)
        component_profit = self._extract_total_profit(component_results)
        full_profit = self._extract_total_profit(full_results)
        
        baseline_win_rate = self._extract_win_rate(baseline_results)
        component_win_rate = self._extract_win_rate(component_results)
        full_win_rate = self._extract_win_rate(full_results)
        
        # Calculate absolute improvement
        profit_improvement = component_profit - baseline_profit
        win_rate_improvement = component_win_rate - baseline_win_rate
        
        # Calculate relative contribution to full ensemble
        if full_profit != baseline_profit:
            profit_contribution = profit_improvement / (full_profit - baseline_profit) * 100
        else:
            profit_contribution = 0
            
        if full_win_rate != baseline_win_rate:
            win_rate_contribution = win_rate_improvement / (full_win_rate - baseline_win_rate) * 100
        else:
            win_rate_contribution = 0
        
        return {
            "profit_improvement": profit_improvement,
            "win_rate_improvement": win_rate_improvement,
            "profit_contribution_pct": profit_contribution,
            "win_rate_contribution_pct": win_rate_contribution
        }
    
    def _extract_total_profit(self, results):
        """
        Extract total profit from results
        
        Args:
            results (dict): Results dictionary
            
        Returns:
            float: Total profit
        """
        total_profit = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_profit += timeframe_results.get('net_profit', 0)
        
        return total_profit
    
    def _extract_win_rate(self, results):
        """
        Extract win rate from results
        
        Args:
            results (dict): Results dictionary
            
        Returns:
            float: Win rate (0-1)
        """
        total_trades = 0
        winning_trades = 0
        
        for symbol, symbol_results in results.items():
            for timeframe, timeframe_results in symbol_results.items():
                if isinstance(timeframe_results, dict) and "error" not in timeframe_results:
                    total_trades += timeframe_results.get('total_trades', 0)
                    winning_trades += timeframe_results.get('winning_trades', 0)
        
        return winning_trades / total_trades if total_trades > 0 else 0

    def run_monte_carlo_simulations(self, period_name="1_year", component_name="full_ensemble", simulation_count=1000):
        """
        Run Monte Carlo simulations with parameter randomization
        
        Args:
            period_name (str): Period to test
            component_name (str): Component to test
            simulation_count (int): Number of simulations to run
            
        Returns:
            dict: Monte Carlo simulation results
        """
        logger.info(f"Running Monte Carlo simulations for period: {period_name}, component: {component_name}")
        logger.info(f"Simulation count: {simulation_count}")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize Monte Carlo simulation results
        mc_results = {
            "simulations": [],
            "parameter_sensitivity": {},
            "risk_profile": {
                "var_95": 0.0,  # Value at Risk (95% confidence)
                "var_99": 0.0,  # Value at Risk (99% confidence)
                "cvar_95": 0.0,  # Conditional Value at Risk (95%)
                "max_drawdown_distribution": [],
                "profit_distribution": [],
                "sharpe_ratio_distribution": [],
                "win_rate_distribution": []
            },
            "failure_points": [],
            "summary": {
                "mean_return": 0.0,
                "median_return": 0.0,
                "std_dev_return": 0.0,
                "mean_sharpe": 0.0,
                "mean_max_drawdown": 0.0,
                "mean_win_rate": 0.0,
                "best_params": {},
                "worst_params": {}
            }
        }
        
        # Parameter ranges for randomization
        param_ranges = {
            # Trading parameters
            "stop_loss": (-30.0, -5.0),
            "take_profit": [(5.0, 20.0), (15.0, 50.0), (30.0, 100.0), (50.0, 200.0)],
            "position_size_multiplier": (1.0, 15.0),
            "max_concurrent_trades": (1, 15),
            "initial_risk_per_trade": (0.01, 0.05),
            "max_risk_per_trade": (0.03, 0.1),
            "max_open_risk": (0.1, 0.4),
            
            # Component weights
            "markov_confidence_weight": (0.1, 0.5),
            "ml_confidence_weight": (0.1, 0.5),
            "fractal_confidence_weight": (0.1, 0.5),
            "price_action_weight": (0.05, 0.3),
            "volume_confidence_weight": (0.05, 0.3),
            
            # Signal thresholds
            "min_confidence": (0.2, 0.6),
            "min_signal_strength": (0.3, 0.7),
            
            # Lookback periods
            "market_regime_lookback": (10, 50),
            "volatility_lookback": (10, 50),
            "support_resistance_lookback": (20, 100)
        }
        
        # Track parameter sensitivity
        param_performance_correlation = {}
        for param in param_ranges.keys():
            param_performance_correlation[param] = []
        
        # Run simulations
        best_return = -float('inf')
        worst_return = float('inf')
        best_params = {}
        worst_params = {}
        
        # Create progress bar
        with tqdm(total=simulation_count, desc="Running Monte Carlo simulations") as pbar:
            for i in range(simulation_count):
                # Randomize parameters
                params = {}
                for param, range_val in param_ranges.items():
                    if param == "take_profit":
                        # Handle take profit levels (list of ranges)
                        tp_levels = []
                        for level_range in range_val:
                            tp_levels.append(random.uniform(level_range[0], level_range[1]))
                        params[param] = sorted(tp_levels)  # Ensure ascending order
                    elif isinstance(range_val, tuple) and len(range_val) == 2:
                        if isinstance(range_val[0], int) and isinstance(range_val[1], int):
                            # Integer parameter
                            params[param] = random.randint(range_val[0], range_val[1])
                        else:
                            # Float parameter
                            params[param] = random.uniform(range_val[0], range_val[1])
                
                # Create backtest instance
                backtest = UltraOptimizedBacktest(
                    symbol=self.symbols[0],  # Use first symbol for simplicity
                    timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                    initial_balance=self.initial_balance,
                    max_position_size=self.max_position_size
                )
                
                # Set up component-specific parameters
                if component_name != "full_ensemble":
                    # Disable all components
                    for comp in ["use_support_resistance", "use_volatility_analysis", 
                                "use_momentum_divergence", "use_order_flow", 
                                "use_market_regime", "use_correlation_analysis",
                                "use_bid_ask_imbalance", "use_hurst_exponent"]:
                        params[comp] = False
                    
                    # Enable only the specified component
                    component_param_map = {
                        "support_resistance": "use_support_resistance",
                        "volatility": "use_volatility_analysis",
                        "momentum_divergence": "use_momentum_divergence",
                        "order_flow": "use_order_flow",
                        "market_regime": "use_market_regime",
                        "correlation": "use_correlation_analysis",
                        "bid_ask_imbalance": "use_bid_ask_imbalance",
                        "hurst_exponent": "use_hurst_exponent"
                    }
                    
                    if component_name in component_param_map:
                        params[component_param_map[component_name]] = True
                else:
                    # Enable all components for full ensemble
                    for comp in ["use_support_resistance", "use_volatility_analysis", 
                                "use_momentum_divergence", "use_order_flow", 
                                "use_market_regime", "use_correlation_analysis",
                                "use_bid_ask_imbalance", "use_hurst_exponent"]:
                        params[comp] = random.random() > 0.3  # 70% chance of enabling each component
                
                # Apply parameters
                backtest.strategy.params.update(params)
                
                # Run backtest
                try:
                    results = backtest.run(
                        start_date=period_info["start_date"],
                        end_date=period_info["end_date"]
                    )
                    
                    # Extract key metrics
                    total_return = results.get("total_profit", 0.0)
                    max_drawdown = results.get("max_drawdown", 0.0)
                    sharpe_ratio = results.get("sharpe_ratio", 0.0)
                    win_rate = results.get("win_rate", 0.0)
                    
                    # Store simulation results
                    sim_result = {
                        "params": params.copy(),
                        "metrics": {
                            "total_return": total_return,
                            "max_drawdown": max_drawdown,
                            "sharpe_ratio": sharpe_ratio,
                            "win_rate": win_rate,
                            "profit_factor": results.get("profit_factor", 0.0),
                            "total_trades": results.get("total_trades", 0)
                        }
                    }
                    
                    mc_results["simulations"].append(sim_result)
                    
                    # Update distributions
                    mc_results["risk_profile"]["max_drawdown_distribution"].append(max_drawdown)
                    mc_results["risk_profile"]["profit_distribution"].append(total_return)
                    mc_results["risk_profile"]["sharpe_ratio_distribution"].append(sharpe_ratio)
                    mc_results["risk_profile"]["win_rate_distribution"].append(win_rate)
                    
                    # Track parameter sensitivity
                    for param, value in params.items():
                        if param in param_performance_correlation:
                            if param == "take_profit":
                                # Use the average take profit for correlation
                                value = sum(value) / len(value)
                            param_performance_correlation[param].append((value, total_return))
                    
                    # Track best and worst parameters
                    if total_return > best_return:
                        best_return = total_return
                        best_params = params.copy()
                    
                    if total_return < worst_return and total_return > -self.initial_balance:
                        worst_return = total_return
                        worst_params = params.copy()
                    
                    # Identify failure points (extreme drawdowns or negative returns)
                    if max_drawdown > 50.0 or total_return < -0.3 * self.initial_balance:
                        mc_results["failure_points"].append({
                            "params": params.copy(),
                            "max_drawdown": max_drawdown,
                            "total_return": total_return
                        })
                
                except Exception as e:
                    logger.error(f"Error in Monte Carlo simulation {i}: {e}")
                
                # Update progress bar
                pbar.update(1)
        
        # Calculate parameter sensitivity
        for param, values in param_performance_correlation.items():
            if values:
                # Calculate correlation between parameter value and return
                param_values = [v[0] for v in values]
                returns = [v[1] for v in values]
                
                if len(set(param_values)) > 1:  # Ensure there's variation in parameter values
                    correlation = np.corrcoef(param_values, returns)[0, 1]
                    mc_results["parameter_sensitivity"][param] = {
                        "correlation": correlation,
                        "effect_size": abs(correlation),
                        "direction": "positive" if correlation > 0 else "negative"
                    }
        
        # Calculate risk metrics
        if mc_results["risk_profile"]["profit_distribution"]:
            profits = sorted(mc_results["risk_profile"]["profit_distribution"])
            drawdowns = sorted(mc_results["risk_profile"]["max_drawdown_distribution"])
            sharpes = sorted(mc_results["risk_profile"]["sharpe_ratio_distribution"])
            win_rates = sorted(mc_results["risk_profile"]["win_rate_distribution"])
            
            # Calculate Value at Risk (VaR)
            var_95_index = int(0.05 * len(profits))
            var_99_index = int(0.01 * len(profits))
            
            mc_results["risk_profile"]["var_95"] = abs(profits[var_95_index]) if var_95_index < len(profits) else 0
            mc_results["risk_profile"]["var_99"] = abs(profits[var_99_index]) if var_99_index < len(profits) else 0
            
            # Calculate Conditional Value at Risk (CVaR)
            cvar_samples = profits[:var_95_index+1]
            mc_results["risk_profile"]["cvar_95"] = abs(sum(cvar_samples) / len(cvar_samples)) if cvar_samples else 0
            
            # Calculate summary statistics
            mc_results["summary"]["mean_return"] = np.mean(profits)
            mc_results["summary"]["median_return"] = np.median(profits)
            mc_results["summary"]["std_dev_return"] = np.std(profits)
            mc_results["summary"]["mean_sharpe"] = np.mean(sharpes)
            mc_results["summary"]["mean_max_drawdown"] = np.mean(drawdowns)
            mc_results["summary"]["mean_win_rate"] = np.mean(win_rates)
            mc_results["summary"]["best_params"] = best_params
            mc_results["summary"]["worst_params"] = worst_params
        
        # Log summary
        logger.info(f"Monte Carlo simulations completed for {component_name} in period {period_name}")
        logger.info(f"Mean return: {mc_results['summary']['mean_return']:.2f}")
        logger.info(f"Mean Sharpe ratio: {mc_results['summary']['mean_sharpe']:.2f}")
        logger.info(f"Mean max drawdown: {mc_results['summary']['mean_max_drawdown']:.2f}%")
        logger.info(f"Value at Risk (95%): {mc_results['risk_profile']['var_95']:.2f}")
        logger.info(f"Number of failure points identified: {len(mc_results['failure_points'])}")
        
        # Store results
        self.monte_carlo_results[f"{period_name}_{component_name}"] = mc_results
        
        # Save results to file
        self._save_monte_carlo_results(mc_results, period_name, component_name)
        
        return mc_results
    
    def _save_monte_carlo_results(self, results, period_name, component_name):
        """
        Save Monte Carlo simulation results to file
        
        Args:
            results (dict): Monte Carlo simulation results
            period_name (str): Period tested
            component_name (str): Component tested
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs("results/monte_carlo_results", exist_ok=True)
        
        # Save results to file
        filename = f"results/monte_carlo_results/monte_carlo_{component_name}_{period_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Monte Carlo simulation results saved to {filename}")
    
    def run_stress_tests(self, period_name="1_year", component_name="full_ensemble"):
        """
        Run stress tests to evaluate strategy performance under extreme market conditions
        
        Args:
            period_name (str): Period to test
            component_name (str): Component to test
            
        Returns:
            dict: Stress test results
        """
        logger.info(f"Running stress tests for period: {period_name}, component: {component_name}")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize stress test results
        stress_results = {
            "scenarios": {},
            "summary": {
                "worst_case_return": 0.0,
                "worst_case_drawdown": 0.0,
                "recovery_time_avg": 0.0,
                "failed_scenarios": [],
                "robust_scenarios": []
            }
        }
        
        # Define stress test scenarios
        scenarios = {
            "flash_crash": {
                "description": "Simulate a flash crash with sudden price drop of 20-30% within minutes",
                "price_modifier": lambda price_data: self._simulate_flash_crash(price_data, drop_pct=25, recovery_pct=15),
                "volume_modifier": lambda volume_data: volume_data * 3.0,  # Triple volume during crash
                "connectivity": "normal"
            },
            "extreme_volatility": {
                "description": "Period of extreme volatility with large price swings",
                "price_modifier": lambda price_data: self._simulate_extreme_volatility(price_data, volatility_multiplier=3.0),
                "volume_modifier": lambda volume_data: volume_data * 2.0,
                "connectivity": "normal"
            },
            "low_liquidity": {
                "description": "Period of low liquidity with wider spreads",
                "price_modifier": lambda price_data: price_data,  # No direct price modification
                "volume_modifier": lambda volume_data: volume_data * 0.3,  # 70% reduction in volume
                "spread_multiplier": 3.0,  # Triple the spread
                "connectivity": "normal"
            },
            "rapid_reversal": {
                "description": "Rapid market reversal after a trend",
                "price_modifier": lambda price_data: self._simulate_rapid_reversal(price_data),
                "volume_modifier": lambda volume_data: volume_data * 1.5,
                "connectivity": "normal"
            },
            "connectivity_issues": {
                "description": "Intermittent connectivity issues causing delayed order execution",
                "price_modifier": lambda price_data: price_data,  # No price modification
                "volume_modifier": lambda volume_data: volume_data,
                "connectivity": "intermittent",
                "execution_delay": (5, 15)  # Random delay between 5-15 seconds
            },
            "data_delay": {
                "description": "Delayed market data feed",
                "price_modifier": lambda price_data: price_data,
                "volume_modifier": lambda volume_data: volume_data,
                "connectivity": "delayed",
                "data_delay": 30  # 30 second delay in data feed
            },
            "price_gaps": {
                "description": "Significant overnight or weekend price gaps",
                "price_modifier": lambda price_data: self._simulate_price_gaps(price_data),
                "volume_modifier": lambda volume_data: volume_data * 1.2,
                "connectivity": "normal"
            },
            "extreme_trend": {
                "description": "Extremely strong trend in one direction",
                "price_modifier": lambda price_data: self._simulate_extreme_trend(price_data, trend_strength=2.0),
                "volume_modifier": lambda volume_data: volume_data * 1.5,
                "connectivity": "normal"
            },
            "choppy_market": {
                "description": "Extremely choppy market with no clear direction",
                "price_modifier": lambda price_data: self._simulate_choppy_market(price_data),
                "volume_modifier": lambda volume_data: volume_data * 0.8,
                "connectivity": "normal"
            },
            "combined_stress": {
                "description": "Combined stress scenario with multiple adverse conditions",
                "price_modifier": lambda price_data: self._simulate_combined_stress(price_data),
                "volume_modifier": lambda volume_data: self._simulate_combined_volume_stress(volume_data),
                "connectivity": "intermittent",
                "execution_delay": (2, 20),
                "spread_multiplier": 2.5
            }
        }
        
        # Run each stress test scenario
        for scenario_name, scenario_config in scenarios.items():
            logger.info(f"Running stress test scenario: {scenario_name} - {scenario_config['description']}")
            
            try:
                # Create backtest instance for this scenario
                backtest = UltraOptimizedBacktest(
                    symbol=self.symbols[0],  # Use first symbol for simplicity
                    timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                    initial_balance=self.initial_balance,
                    max_position_size=self.max_position_size
                )
                
                # Set up component-specific parameters
                params = self._get_component_params(component_name)
                
                # Apply parameters
                backtest.strategy.params.update(params)
                
                # Configure stress test
                backtest.strategy.stress_test_config = {
                    "enabled": True,
                    "scenario": scenario_name,
                    "price_modifier": scenario_config["price_modifier"],
                    "volume_modifier": scenario_config["volume_modifier"],
                    "connectivity": scenario_config["connectivity"]
                }
                
                # Add additional stress test parameters
                if "spread_multiplier" in scenario_config:
                    backtest.strategy.stress_test_config["spread_multiplier"] = scenario_config["spread_multiplier"]
                
                if "execution_delay" in scenario_config:
                    backtest.strategy.stress_test_config["execution_delay"] = scenario_config["execution_delay"]
                
                if "data_delay" in scenario_config:
                    backtest.strategy.stress_test_config["data_delay"] = scenario_config["data_delay"]
                
                # Run backtest
                results = backtest.run(
                    start_date=period_info["start_date"],
                    end_date=period_info["end_date"]
                )
                
                # Calculate recovery metrics if available
                recovery_time = self._calculate_recovery_time(results)
                
                # Store results
                stress_results["scenarios"][scenario_name] = {
                    "description": scenario_config["description"],
                    "metrics": {
                        "total_return": results.get("total_profit", 0.0),
                        "max_drawdown": results.get("max_drawdown", 0.0),
                        "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                        "win_rate": results.get("win_rate", 0.0),
                        "profit_factor": results.get("profit_factor", 0.0),
                        "total_trades": results.get("total_trades", 0),
                        "recovery_time": recovery_time
                    }
                }
                
                # Update summary
                if results.get("total_profit", 0.0) < stress_results["summary"]["worst_case_return"]:
                    stress_results["summary"]["worst_case_return"] = results.get("total_profit", 0.0)
                
                if results.get("max_drawdown", 0.0) > stress_results["summary"]["worst_case_drawdown"]:
                    stress_results["summary"]["worst_case_drawdown"] = results.get("max_drawdown", 0.0)
                
                # Classify scenario as failed or robust
                if results.get("total_profit", 0.0) < -0.15 * self.initial_balance or results.get("max_drawdown", 0.0) > 40.0:
                    stress_results["summary"]["failed_scenarios"].append(scenario_name)
                elif results.get("total_profit", 0.0) > 0 and results.get("max_drawdown", 0.0) < 25.0:
                    stress_results["summary"]["robust_scenarios"].append(scenario_name)
                
            except Exception as e:
                logger.error(f"Error in stress test scenario {scenario_name}: {e}")
                stress_results["scenarios"][scenario_name] = {
                    "description": scenario_config["description"],
                    "error": str(e)
                }
        
        # Calculate average recovery time
        recovery_times = [
            scenario["metrics"]["recovery_time"] 
            for scenario_name, scenario in stress_results["scenarios"].items() 
            if "metrics" in scenario and "recovery_time" in scenario["metrics"] and scenario["metrics"]["recovery_time"] > 0
        ]
        
        if recovery_times:
            stress_results["summary"]["recovery_time_avg"] = sum(recovery_times) / len(recovery_times)
        
        # Log summary
        logger.info(f"Stress tests completed for {component_name} in period {period_name}")
        logger.info(f"Worst case return: {stress_results['summary']['worst_case_return']:.2f}")
        logger.info(f"Worst case drawdown: {stress_results['summary']['worst_case_drawdown']:.2f}%")
        logger.info(f"Average recovery time: {stress_results['summary']['recovery_time_avg']:.2f} days")
        logger.info(f"Failed scenarios: {len(stress_results['summary']['failed_scenarios'])}")
        logger.info(f"Robust scenarios: {len(stress_results['summary']['robust_scenarios'])}")
        
        # Store results
        self.stress_test_results[f"{period_name}_{component_name}"] = stress_results
        
        # Save results to file
        self._save_stress_test_results(stress_results, period_name, component_name)
        
        return stress_results
    
    def _simulate_flash_crash(self, price_data, drop_pct=25, recovery_pct=15):
        """
        Simulate a flash crash in the price data
        
        Args:
            price_data (pd.DataFrame): Price data
            drop_pct (float): Percentage drop during crash
            recovery_pct (float): Percentage recovery after crash
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Make a copy of the data to avoid modifying the original
        modified_data = price_data.copy()
        
        # Choose a random point for the flash crash (not too early or late)
        crash_idx = random.randint(int(len(modified_data) * 0.3), int(len(modified_data) * 0.7))
        
        # Define crash duration (in candles)
        crash_duration = random.randint(5, 15)
        recovery_duration = random.randint(10, 30)
        
        # Calculate drop and recovery factors
        drop_factor = 1.0 - (drop_pct / 100.0)
        recovery_factor = 1.0 + (recovery_pct / 100.0)
        
        # Apply flash crash
        for i in range(crash_duration):
            if crash_idx + i < len(modified_data):
                # Progressive crash
                crash_progress = (i + 1) / crash_duration
                current_drop = 1.0 - (crash_progress * (1.0 - drop_factor))
                
                modified_data.iloc[crash_idx + i, :] = modified_data.iloc[crash_idx + i, :] * current_drop
        
        # Apply recovery
        for i in range(recovery_duration):
            if crash_idx + crash_duration + i < len(modified_data):
                # Progressive recovery
                recovery_progress = (i + 1) / recovery_duration
                current_recovery = drop_factor + (recovery_progress * (recovery_factor - drop_factor))
                
                modified_data.iloc[crash_idx + crash_duration + i, :] = modified_data.iloc[crash_idx + crash_duration + i, :] * current_recovery
        
        return modified_data
    
    def _simulate_extreme_volatility(self, price_data, volatility_multiplier=3.0):
        """
        Simulate extreme volatility in the price data
        
        Args:
            price_data (pd.DataFrame): Price data
            volatility_multiplier (float): Multiplier for volatility
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Make a copy of the data to avoid modifying the original
        modified_data = price_data.copy()
        
        # Choose a random segment for extreme volatility
        start_idx = random.randint(int(len(modified_data) * 0.2), int(len(modified_data) * 0.6))
        duration = random.randint(30, 100)
        end_idx = min(start_idx + duration, len(modified_data))
        
        # Calculate historical volatility
        returns = np.log(modified_data['close'] / modified_data['close'].shift(1))
        historical_volatility = returns.rolling(window=20).std().mean()
        
        # Generate random volatility shocks
        for i in range(start_idx, end_idx):
            # Random volatility shock
            shock = np.random.normal(0, historical_volatility * volatility_multiplier)
            
            # Apply shock to prices
            shock_factor = np.exp(shock)
            modified_data.iloc[i, :] = modified_data.iloc[i, :] * shock_factor
        
        return modified_data
    
    def _simulate_rapid_reversal(self, price_data):
        """
        Simulate a rapid market reversal
        
        Args:
            price_data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Make a copy of the data to avoid modifying the original
        modified_data = price_data.copy()
        
        # Determine if the market is in an uptrend or downtrend
        window = min(50, len(modified_data) // 4)
        start_price = modified_data['close'].iloc[0:window].mean()
        end_price = modified_data['close'].iloc[-window:].mean()
        
        is_uptrend = end_price > start_price
        
        # Choose a point for the reversal (in the last third)
        reversal_idx = random.randint(int(len(modified_data) * 0.6), int(len(modified_data) * 0.8))
        
        # Calculate reversal factor
        if is_uptrend:
            # Downward reversal
            reversal_factor = 0.85  # 15% drop
        else:
            # Upward reversal
            reversal_factor = 1.15  # 15% rise
        
        # Apply reversal
        for i in range(reversal_idx, len(modified_data)):
            # Progressive reversal
            progress = min(1.0, (i - reversal_idx) / 20)  # Full effect over 20 candles
            current_factor = 1.0 + progress * (reversal_factor - 1.0)
            
            modified_data.iloc[i, :] = modified_data.iloc[i, :] * current_factor
        
        return modified_data
    
    def _simulate_price_gaps(self, price_data):
        """
        Simulate significant price gaps
        
        Args:
            price_data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Make a copy of the data to avoid modifying the original
        modified_data = price_data.copy()
        
        # Number of gaps to introduce
        num_gaps = random.randint(3, 8)
        
        # Minimum distance between gaps
        min_gap_distance = len(modified_data) // (num_gaps * 2)
        
        # Create gaps
        gap_indices = []
        for _ in range(num_gaps):
            # Find a suitable position for the gap
            while True:
                gap_idx = random.randint(10, len(modified_data) - 10)
                
                # Check if it's far enough from other gaps
                if all(abs(gap_idx - existing_idx) > min_gap_distance for existing_idx in gap_indices):
                    gap_indices.append(gap_idx)
                    break
        
        # Apply gaps
        for gap_idx in gap_indices:
            # Determine gap direction and size
            gap_direction = 1 if random.random() > 0.5 else -1
            gap_size = random.uniform(0.03, 0.08)  # 3-8% gap
            
            # Apply gap
            gap_factor = 1.0 + (gap_direction * gap_size)
            
            # Apply to all prices after the gap
            for i in range(gap_idx, len(modified_data)):
                modified_data.iloc[i, :] = modified_data.iloc[i, :] * gap_factor
        
        return modified_data
    
    def _simulate_extreme_trend(self, price_data, trend_strength=2.0):
        """
        Simulate an extremely strong trend
        
        Args:
            price_data (pd.DataFrame): Price data
            trend_strength (float): Strength multiplier for the trend
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Make a copy of the data to avoid modifying the original
        modified_data = price_data.copy()
        
        # Determine trend direction (random)
        trend_direction = 1 if random.random() > 0.5 else -1
        
        # Calculate baseline trend
        start_price = modified_data['close'].iloc[0]
        end_price = modified_data['close'].iloc[-1]
        natural_trend = (end_price / start_price) - 1.0
        
        # Strengthen the trend
        enhanced_trend = natural_trend * trend_strength
        
        # Apply trend
        for i in range(len(modified_data)):
            # Progressive trend application
            progress = i / len(modified_data)
            trend_factor = 1.0 + (progress * enhanced_trend * trend_direction)
            
            modified_data.iloc[i, :] = modified_data.iloc[i, :] * trend_factor
        
        return modified_data
    
    def _simulate_choppy_market(self, price_data):
        """
        Simulate a choppy market with no clear direction
        
        Args:
            price_data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Make a copy of the data to avoid modifying the original
        modified_data = price_data.copy()
        
        # Calculate baseline trend and remove it
        start_price = modified_data['close'].iloc[0]
        end_price = modified_data['close'].iloc[-1]
        trend_factor = (end_price / start_price) ** (1 / len(modified_data))
        
        # Remove the trend
        for i in range(len(modified_data)):
            modified_data.iloc[i, :] = modified_data.iloc[i, :] / (trend_factor ** i)
        
        # Add choppy noise
        for i in range(len(modified_data)):
            # Random noise
            noise = np.random.normal(0, 0.015)  # 1.5% standard deviation
            
            # Apply noise
            noise_factor = 1.0 + noise
            modified_data.iloc[i, :] = modified_data.iloc[i, :] * noise_factor
        
        return modified_data
    
    def _simulate_combined_stress(self, price_data):
        """
        Simulate combined stress conditions
        
        Args:
            price_data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Modified price data
        """
        # Apply multiple stress scenarios in sequence
        modified_data = self._simulate_extreme_volatility(price_data, volatility_multiplier=2.0)
        modified_data = self._simulate_flash_crash(modified_data, drop_pct=15, recovery_pct=10)
        modified_data = self._simulate_price_gaps(modified_data)
        
        return modified_data
    
    def _simulate_combined_volume_stress(self, volume_data):
        """
        Simulate combined volume stress conditions
        
        Args:
            volume_data (pd.Series): Volume data
            
        Returns:
            pd.Series: Modified volume data
        """
        # Make a copy of the data
        modified_volume = volume_data.copy()
        
        # Apply volume spikes and drops
        for i in range(len(modified_volume)):
            if random.random() < 0.1:  # 10% chance of volume anomaly
                if random.random() < 0.7:  # 70% chance of spike, 30% chance of drop
                    # Volume spike
                    spike_factor = random.uniform(2.0, 5.0)
                    modified_volume.iloc[i] = modified_volume.iloc[i] * spike_factor
                else:
                    # Volume drop
                    drop_factor = random.uniform(0.1, 0.5)
                    modified_volume.iloc[i] = modified_volume.iloc[i] * drop_factor
        
        return modified_volume
    
    def _calculate_recovery_time(self, results):
        """
        Calculate recovery time from maximum drawdown
        
        Args:
            results (dict): Backtest results
            
        Returns:
            float: Recovery time in days
        """
        if "equity_curve" not in results or "max_drawdown_end_idx" not in results:
            return 0.0
        
        equity_curve = results["equity_curve"]
        max_dd_end_idx = results["max_drawdown_end_idx"]
        
        if max_dd_end_idx >= len(equity_curve) - 1:
            # No recovery if drawdown ends at the last point
            return float('inf')
        
        # Find peak value before drawdown
        max_dd_start_idx = results.get("max_drawdown_start_idx", 0)
        peak_value = equity_curve[max_dd_start_idx]
        
        # Find recovery point
        recovery_idx = max_dd_end_idx
        for i in range(max_dd_end_idx + 1, len(equity_curve)):
            if equity_curve[i] >= peak_value:
                recovery_idx = i
                break
        
        # Calculate recovery time (assuming daily data)
        recovery_time = recovery_idx - max_dd_end_idx
        
        return recovery_time
    
    def _get_component_params(self, component_name):
        """
        Get parameters for a specific component
        
        Args:
            component_name (str): Component name
            
        Returns:
            dict: Component parameters
        """
        # Base parameters
        params = {
            # Trading parameters
            "stop_loss": -15.0,
            "take_profit": [10.0, 25.0, 50.0, 100.0],
            "position_size_multiplier": 5.0,
            "max_concurrent_trades": 5,
            "initial_risk_per_trade": 0.02,
            "max_risk_per_trade": 0.05,
            "max_open_risk": 0.2,
            
            # Disable all components by default
            "use_support_resistance": False,
            "use_volatility_analysis": False,
            "use_momentum_divergence": False,
            "use_order_flow": False,
            "use_market_regime": False,
            "use_correlation_analysis": False,
            "use_bid_ask_imbalance": False,
            "use_hurst_exponent": False
        }
        
        # Enable specific component
        if component_name != "full_ensemble":
            component_param_map = {
                "support_resistance": "use_support_resistance",
                "volatility": "use_volatility_analysis",
                "momentum_divergence": "use_momentum_divergence",
                "order_flow": "use_order_flow",
                "market_regime": "use_market_regime",
                "correlation": "use_correlation_analysis",
                "bid_ask_imbalance": "use_bid_ask_imbalance",
                "hurst_exponent": "use_hurst_exponent"
            }
            
            if component_name in component_param_map:
                params[component_param_map[component_name]] = True
        else:
            # Enable all components for full ensemble
            for comp in ["use_support_resistance", "use_volatility_analysis", 
                        "use_momentum_divergence", "use_order_flow", 
                        "use_market_regime", "use_correlation_analysis",
                        "use_bid_ask_imbalance", "use_hurst_exponent"]:
                params[comp] = True
        
        return params
    
    def run_cross_validation(self, period_name="1_year", component_name="full_ensemble", folds=5):
        """
        Run cross-validation to prevent overfitting
        
        Args:
            period_name (str): Period to test
            component_name (str): Component to test
            folds (int): Number of folds for cross-validation
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Running cross-validation for period: {period_name}, component: {component_name}")
        logger.info(f"Using {folds} folds")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize cross-validation results
        cv_results = {
            "folds": [],
            "walk_forward_windows": [],
            "nested_cv_results": [],
            "multi_timeframe_results": {},
            "summary": {
                "mean_return": 0.0,
                "median_return": 0.0,
                "std_dev_return": 0.0,
                "mean_sharpe": 0.0,
                "mean_win_rate": 0.0,
                "consistency_score": 0.0,
                "overfitting_score": 0.0,
                "robustness_score": 0.0,
                "multi_timeframe_consistency": 0.0,
                "parameter_sensitivity": 0.0,
                "regime_stability": 0.0,
                "statistical_significance": 0.0
            }
        }
        
        # Get data for the period
        symbol = self.symbols[0]  # Use first symbol for simplicity
        
        try:
            # Create backtest instance to load data
            backtest = UltraOptimizedBacktest(
                symbol=symbol,
                timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                initial_balance=self.initial_balance,
                max_position_size=self.max_position_size
            )
            
            # Load data for the period
            data = backtest.load_data(
                start_date=period_info["start_date"],
                end_date=period_info["end_date"]
            )
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for period {period_name}")
                return {}
            
            # Run k-fold cross-validation
            fold_results = self._run_k_fold_cross_validation(
                data, component_name, folds
            )
            
            # Run walk-forward analysis
            walk_forward_results = self._run_walk_forward_analysis(
                data, component_name
            )
            
            # Run nested cross-validation for hyperparameter tuning
            nested_cv_results = self._run_nested_cross_validation(
                data, component_name, folds
            )
            
            # Store results
            cv_results["folds"] = fold_results
            cv_results["walk_forward_windows"] = walk_forward_results
            cv_results["nested_cv_results"] = nested_cv_results
            
            # Run multi-timeframe robustness testing
            multi_tf_results = self._run_multi_timeframe_robustness_test(
                symbol, period_info, component_name
            )
            cv_results["multi_timeframe_results"] = multi_tf_results
            
            # Calculate summary statistics
            fold_returns = [fold["test_metrics"]["total_return"] for fold in fold_results]
            fold_sharpes = [fold["test_metrics"]["sharpe_ratio"] for fold in fold_results]
            fold_win_rates = [fold["test_metrics"]["win_rate"] for fold in fold_results]
            
            wf_returns = [window["test_metrics"]["total_return"] for window in walk_forward_results]
            wf_sharpes = [window["test_metrics"]["sharpe_ratio"] for window in walk_forward_results]
            wf_win_rates = [window["test_metrics"]["win_rate"] for window in walk_forward_results]
            
            # Calculate mean metrics
            cv_results["summary"]["mean_return"] = np.mean(fold_returns + wf_returns)
            cv_results["summary"]["std_dev_return"] = np.std(fold_returns + wf_returns)
            cv_results["summary"]["mean_sharpe"] = np.mean(fold_sharpes + wf_sharpes)
            cv_results["summary"]["mean_win_rate"] = np.mean(fold_win_rates + wf_win_rates)
            
            # Calculate consistency score (higher is better)
            # Based on coefficient of variation (lower CV means more consistent)
            returns_cv = np.std(fold_returns + wf_returns) / abs(np.mean(fold_returns + wf_returns)) if np.mean(fold_returns + wf_returns) != 0 else float('inf')
            cv_results["summary"]["consistency_score"] = max(0, 100 * (1 - min(returns_cv, 1)))
            
            # Calculate overfitting score (lower is better)
            # Based on difference between in-sample and out-of-sample performance
            train_test_diffs = []
            for fold in fold_results:
                train_return = fold["train_metrics"]["total_return"]
                test_return = fold["test_metrics"]["total_return"]
                if train_return != 0:
                    train_test_diffs.append(abs((train_return - test_return) / train_return))
            
            for window in walk_forward_results:
                train_return = window["train_metrics"]["total_return"]
                test_return = window["test_metrics"]["total_return"]
                if train_return != 0:
                    train_test_diffs.append(abs((train_return - test_return) / train_return))
            
            avg_diff = np.mean(train_test_diffs) if train_test_diffs else 0
            cv_results["summary"]["overfitting_score"] = min(100, 100 * avg_diff)
            
            # Calculate robustness score (higher is better)
            # Based on percentage of profitable folds/windows
            profitable_count = sum(1 for r in fold_returns + wf_returns if r > 0)
            total_count = len(fold_returns + wf_returns)
            cv_results["summary"]["robustness_score"] = (profitable_count / total_count * 100) if total_count > 0 else 0
            
            # Calculate multi-timeframe consistency (higher is better)
            # Based on consistency of returns across different timeframes
            if multi_tf_results and "timeframe_returns" in multi_tf_results:
                tf_returns = list(multi_tf_results["timeframe_returns"].values())
                if tf_returns:
                    tf_returns_cv = np.std(tf_returns) / abs(np.mean(tf_returns)) if np.mean(tf_returns) != 0 else float('inf')
                    cv_results["summary"]["multi_timeframe_consistency"] = max(0, 100 * (1 - min(tf_returns_cv, 1)))
            
            # Calculate parameter sensitivity (lower is better)
            # Based on variation in performance due to parameter changes in nested CV
            if nested_cv_results and "parameter_variations" in nested_cv_results:
                param_returns = nested_cv_results["parameter_variations"]
                if param_returns:
                    param_returns_cv = np.std(param_returns) / abs(np.mean(param_returns)) if np.mean(param_returns) != 0 else float('inf')
                    cv_results["summary"]["parameter_sensitivity"] = min(100, 100 * min(param_returns_cv, 1))
            
            # Calculate regime stability (higher is better)
            # Based on consistency across different market regimes
            regime_returns = []
            for regime in ["bull_market", "bear_market", "sideways_market"]:
                if regime in self.test_periods:
                    regime_data = backtest.load_data(
                        start_date=self.test_periods[regime]["start_date"],
                        end_date=self.test_periods[regime]["end_date"]
                    )
                    if regime_data is not None and len(regime_data) > 0:
                        params = self._get_component_params(component_name)
                        regime_results = self._run_backtest_on_data(regime_data, params)
                        if regime_results and "total_profit" in regime_results:
                            regime_returns.append(regime_results["total_profit"])
            
            if regime_returns:
                regime_returns_cv = np.std(regime_returns) / abs(np.mean(regime_returns)) if np.mean(regime_returns) != 0 else float('inf')
                cv_results["summary"]["regime_stability"] = max(0, 100 * (1 - min(regime_returns_cv, 1)))
            
            # Calculate statistical significance using t-test
            # Null hypothesis: mean return = 0
            if fold_returns + wf_returns:
                from scipy import stats
                t_stat, p_value = stats.ttest_1samp(fold_returns + wf_returns, 0)
                
                # Convert to percentage, higher is better
                cv_results["summary"]["statistical_significance"] = (1 - p_value) * 100  
            
            # Log summary
            logger.info(f"Cross-validation completed for {component_name} in period {period_name}")
            logger.info(f"Mean return: {cv_results['summary']['mean_return']:.2f}")
            logger.info(f"Return standard deviation: {cv_results['summary']['std_dev_return']:.2f}")
            logger.info(f"Mean Sharpe ratio: {cv_results['summary']['mean_sharpe']:.2f}")
            logger.info(f"Consistency score: {cv_results['summary']['consistency_score']:.2f}/100")
            logger.info(f"Overfitting score: {cv_results['summary']['overfitting_score']:.2f}/100 (lower is better)")
            logger.info(f"Robustness score: {cv_results['summary']['robustness_score']:.2f}%")
            logger.info(f"Multi-timeframe consistency: {cv_results['summary']['multi_timeframe_consistency']:.2f}/100")
            logger.info(f"Parameter sensitivity: {cv_results['summary']['parameter_sensitivity']:.2f}/100 (lower is better)")
            logger.info(f"Regime stability: {cv_results['summary']['regime_stability']:.2f}/100")
            logger.info(f"Statistical significance: {cv_results['summary']['statistical_significance']:.2f}%")
            
            # Store results
            self.cross_validation_results[f"{period_name}_{component_name}"] = cv_results
            
            # Save results to file
            self._save_cross_validation_results(cv_results, period_name, component_name)
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def _run_nested_cross_validation(self, data, component_name, folds):
        """
        Run nested cross-validation for hyperparameter tuning
        
        Args:
            data (pd.DataFrame): Price data
            component_name (str): Component to test
            folds (int): Number of folds
            
        Returns:
            dict: Nested cross-validation results
        """
        logger.info(f"Running nested cross-validation for {component_name}")
        
        # Initialize results
        nested_cv_results = {
            "outer_folds": [],
            "best_parameters": {},
            "parameter_variations": []
        }
        
        # Define parameter variations to test
        parameter_variations = {
            "stop_loss": [-10.0, -15.0, -20.0],
            "take_profit": [[10.0, 20.0], [15.0, 30.0], [20.0, 40.0]],
            "position_size_multiplier": [3.0, 5.0, 7.0],
            "max_concurrent_trades": [3, 5, 7]
        }
        
        # Calculate fold size
        fold_size = len(data) // folds
        
        # Run outer cross-validation loop
        for i in range(folds):
            logger.info(f"Processing outer fold {i+1}/{folds}")
            
            # Calculate fold indices for outer loop
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < folds - 1 else len(data)
            
            # Split data for outer loop
            train_data = pd.concat([data.iloc[:test_start], data.iloc[test_end:]])
            test_data = data.iloc[test_start:test_end]
            
            # Skip if either dataset is too small
            if len(train_data) < 200 or len(test_data) < 50:
                logger.warning(f"Skipping outer fold {i+1} due to insufficient data")
                continue
            
            # Initialize inner fold results
            inner_fold_results = []
            
            # Run inner cross-validation loop for hyperparameter tuning
            inner_fold_size = len(train_data) // (folds - 1)
            
            # Test different parameter combinations
            best_params = None
            best_score = float('-inf')
            
            # Track parameter performance variations
            param_performances = []
            
            # Test each parameter combination
            for stop_loss in parameter_variations["stop_loss"]:
                for take_profit in parameter_variations["take_profit"]:
                    for position_size in parameter_variations["position_size_multiplier"]:
                        for max_trades in parameter_variations["max_concurrent_trades"]:
                            # Create parameter set
                            params = self._get_component_params(component_name)
                            params.update({
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "position_size_multiplier": position_size,
                                "max_concurrent_trades": max_trades
                            })
                            
                            # Evaluate on inner folds
                            inner_scores = []
                            
                            for j in range(folds - 1):
                                # Calculate inner fold indices
                                inner_test_start = j * inner_fold_size
                                inner_test_end = (j + 1) * inner_fold_size if j < folds - 2 else len(train_data)
                                
                                # Split inner training data
                                inner_train = pd.concat([train_data.iloc[:inner_test_start], train_data.iloc[inner_test_end:]])
                                inner_test = train_data.iloc[inner_test_start:inner_test_end]
                                
                                if len(inner_train) < 100 or len(inner_test) < 50:
                                    continue
                                
                                # Run backtest on inner fold
                                inner_train_results = self._run_backtest_on_data(inner_train, params)
                                inner_test_results = self._run_backtest_on_data(inner_test, params)
                                
                                # Calculate score (using Sharpe ratio)
                                inner_scores.append(inner_test_results.get("sharpe_ratio", 0.0))
                            
                            # Calculate average score across inner folds
                            avg_score = np.mean(inner_scores) if inner_scores else 0.0
                            
                            # Track parameter performance
                            param_performances.append(avg_score)
                            
                            # Update best parameters if better score
                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = params.copy()
                                
                                # Store inner fold result
                                inner_fold_results.append({
                                    "params": {
                                        "stop_loss": stop_loss,
                                        "take_profit": take_profit,
                                        "position_size_multiplier": position_size,
                                        "max_concurrent_trades": max_trades
                                    },
                                    "score": avg_score
                                })
            
            # Use best parameters from inner CV to evaluate on outer test fold
            if best_params:
                outer_test_results = self._run_backtest_on_data(test_data, best_params)
                
                # Store outer fold result
                nested_cv_results["outer_folds"].append({
                    "fold": i + 1,
                    "best_params": {
                        "stop_loss": best_params.get("stop_loss"),
                        "take_profit": best_params.get("take_profit"),
                        "position_size_multiplier": best_params.get("position_size_multiplier"),
                        "max_concurrent_trades": best_params.get("max_concurrent_trades")
                    },
                    "inner_cv_score": best_score,
                    "test_metrics": {
                        "total_return": outer_test_results.get("total_profit", 0.0),
                        "max_drawdown": outer_test_results.get("max_drawdown", 0.0),
                        "sharpe_ratio": outer_test_results.get("sharpe_ratio", 0.0),
                        "win_rate": outer_test_results.get("win_rate", 0.0)
                    }
                })
        
        # Store parameter performance variations
        nested_cv_results["parameter_variations"] = param_performances
        
        # Calculate overall best parameters
        if nested_cv_results["outer_folds"]:
            # Find most frequently selected parameters
            stop_loss_values = [fold["best_params"]["stop_loss"] for fold in nested_cv_results["outer_folds"]]
            take_profit_values = [str(fold["best_params"]["take_profit"]) for fold in nested_cv_results["outer_folds"]]
            position_size_values = [fold["best_params"]["position_size_multiplier"] for fold in nested_cv_results["outer_folds"]]
            max_trades_values = [fold["best_params"]["max_concurrent_trades"] for fold in nested_cv_results["outer_folds"]]
            
            from collections import Counter
            
            stop_loss_counter = Counter(stop_loss_values)
            take_profit_counter = Counter(take_profit_values)
            position_size_counter = Counter(position_size_values)
            max_trades_counter = Counter(max_trades_values)
            
            # Get most common values
            most_common_stop_loss = stop_loss_counter.most_common(1)[0][0] if stop_loss_counter else -15.0
            most_common_take_profit_str = take_profit_counter.most_common(1)[0][0] if take_profit_counter else "[10.0, 25.0, 50.0, 100.0]"
            most_common_position_size = position_size_counter.most_common(1)[0][0] if position_size_counter else 5.0
            most_common_max_trades = max_trades_counter.most_common(1)[0][0] if max_trades_counter else 5
            
            # Parse take_profit back to list
            import ast
            most_common_take_profit = ast.literal_eval(most_common_take_profit_str) if most_common_take_profit_str.startswith("[") else [10.0, 25.0, 50.0, 100.0]
            
            # Store best parameters
            nested_cv_results["best_parameters"] = {
                "stop_loss": most_common_stop_loss,
                "take_profit": most_common_take_profit,
                "position_size_multiplier": most_common_position_size,
                "max_concurrent_trades": most_common_max_trades
            }
        
        logger.info(f"Nested cross-validation completed with {len(nested_cv_results['outer_folds'])} valid outer folds")
        if "best_parameters" in nested_cv_results:
            logger.info(f"Best parameters: {nested_cv_results['best_parameters']}")
        
        return nested_cv_results

    def _run_multi_timeframe_robustness_test(self, symbol, period_info, component_name):
        """
        Run robustness tests across multiple timeframes
        
        Args:
            symbol (str): Symbol to test
            period_info (dict): Period information
            component_name (str): Component to test
            
        Returns:
            dict: Multi-timeframe test results
        """
        logger.info(f"Running multi-timeframe robustness test for {component_name}")
        
        # Initialize results
        multi_tf_results = {
            "timeframe_results": {},
            "timeframe_returns": {},
            "timeframe_sharpes": {},
            "timeframe_drawdowns": {},
            "timeframe_win_rates": {},
            "correlation_matrix": {},
            "summary": {
                "consistency_score": 0.0,
                "best_timeframe": "",
                "worst_timeframe": ""
            }
        }
        
        # Get parameters for the component
        params = self._get_component_params(component_name)
        
        # Test each timeframe
        for timeframe in self.timeframes:
            logger.info(f"Testing timeframe: {timeframe}")
            
            try:
                # Create backtest instance
                backtest = UltraOptimizedBacktest(
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_balance=self.initial_balance,
                    max_position_size=self.max_position_size
                )
                
                # Load data for the period
                data = backtest.load_data(
                    start_date=period_info["start_date"],
                    end_date=period_info["end_date"]
                )
                
                if data is None or len(data) == 0:
                    logger.warning(f"No data available for timeframe {timeframe}")
                    continue
                
                # Run backtest
                results = self._run_backtest_on_data(data, params)
                
                # Store results
                multi_tf_results["timeframe_results"][timeframe] = results
                multi_tf_results["timeframe_returns"][timeframe] = results.get("total_profit", 0.0)
                multi_tf_results["timeframe_sharpes"][timeframe] = results.get("sharpe_ratio", 0.0)
                multi_tf_results["timeframe_drawdowns"][timeframe] = results.get("max_drawdown", 0.0)
                multi_tf_results["timeframe_win_rates"][timeframe] = results.get("win_rate", 0.0)
                
                logger.info(f"Timeframe {timeframe} - Return: {results.get('total_profit', 0.0):.2f}, Sharpe: {results.get('sharpe_ratio', 0.0):.2f}")
                
            except Exception as e:
                logger.error(f"Error testing timeframe {timeframe}: {e}")
        
        # Calculate correlation between timeframe returns
        if len(multi_tf_results["timeframe_returns"]) > 1:
            # Create correlation matrix
            import pandas as pd
            
            # Create DataFrame with equity curves from each timeframe
            equity_curves = {}
            for timeframe, results in multi_tf_results["timeframe_results"].items():
                if "equity_curve" in results:
                    equity_curves[timeframe] = results["equity_curve"]
            
            if equity_curves:
                # Resample equity curves to daily frequency for comparison
                daily_equity = {}
                for timeframe, curve in equity_curves.items():
                    if isinstance(curve, list) and len(curve) > 0:
                        # Convert to pandas Series if it's a list
                        curve_series = pd.Series(curve)
                        # Resample to daily by taking last value of each day
                        daily_equity[timeframe] = curve_series.resample('D').last()
                
                if daily_equity:
                    # Create DataFrame with all equity curves
                    equity_df = pd.DataFrame(daily_equity)
                    
                    # Calculate correlation matrix
                    corr_matrix = equity_df.corr()
                    
                    # Store correlation matrix
                    multi_tf_results["correlation_matrix"] = corr_matrix.to_dict()
        
        # Calculate summary statistics
        if multi_tf_results["timeframe_returns"]:
            # Calculate consistency score
            returns = list(multi_tf_results["timeframe_returns"].values())
            returns_cv = np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
            multi_tf_results["summary"]["consistency_score"] = max(0, 100 * (1 - min(returns_cv, 1)))
            
            # Find best and worst timeframes
            best_timeframe = max(multi_tf_results["timeframe_returns"].items(), key=lambda x: x[1])
            worst_timeframe = min(multi_tf_results["timeframe_returns"].items(), key=lambda x: x[1])
            
            multi_tf_results["summary"]["best_timeframe"] = best_timeframe[0]
            multi_tf_results["summary"]["worst_timeframe"] = worst_timeframe[0]
            
            logger.info(f"Multi-timeframe consistency score: {multi_tf_results['summary']['consistency_score']:.2f}/100")
            logger.info(f"Best timeframe: {best_timeframe[0]} (Return: {best_timeframe[1]:.2f})")
            logger.info(f"Worst timeframe: {worst_timeframe[0]} (Return: {worst_timeframe[1]:.2f})")
        
        return multi_tf_results
        
    def _run_k_fold_cross_validation(self, data, component_name, folds):
        """
        Run k-fold cross-validation
        
        Args:
            data (pd.DataFrame): Price data
            component_name (str): Component to test
            folds (int): Number of folds
            
        Returns:
            list: Results for each fold
        """
        logger.info(f"Running {folds}-fold cross-validation")
        
        # Initialize results
        fold_results = []
        
        # Calculate fold size
        fold_size = len(data) // folds
        
        # Run each fold
        for i in range(folds):
            logger.info(f"Processing fold {i+1}/{folds}")
            
            # Calculate fold indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < folds - 1 else len(data)
            
            # Split data
            train_data = pd.concat([data.iloc[:test_start], data.iloc[test_end:]])
            test_data = data.iloc[test_start:test_end]
            
            # Skip if either dataset is too small
            if len(train_data) < 100 or len(test_data) < 50:
                logger.warning(f"Skipping fold {i+1} due to insufficient data")
                continue
            
            # Get component parameters
            params = self._get_component_params(component_name)
            
            # Train on training data
            train_results = self._run_backtest_on_data(train_data, params)
            
            # Test on test data
            test_results = self._run_backtest_on_data(test_data, params)
            
            # Store fold results
            fold_results.append({
                "fold": i + 1,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "train_metrics": {
                    "total_return": train_results.get("total_profit", 0.0),
                    "max_drawdown": train_results.get("max_drawdown", 0.0),
                    "sharpe_ratio": train_results.get("sharpe_ratio", 0.0),
                    "win_rate": train_results.get("win_rate", 0.0),
                    "profit_factor": train_results.get("profit_factor", 0.0),
                    "total_trades": train_results.get("total_trades", 0)
                },
                "test_metrics": {
                    "total_return": test_results.get("total_profit", 0.0),
                    "max_drawdown": test_results.get("max_drawdown", 0.0),
                    "sharpe_ratio": test_results.get("sharpe_ratio", 0.0),
                    "win_rate": test_results.get("win_rate", 0.0),
                    "profit_factor": test_results.get("profit_factor", 0.0),
                    "total_trades": test_results.get("total_trades", 0)
                }
            })
            
            logger.info(f"Fold {i+1} - Train return: {train_results.get('total_profit', 0.0):.2f}, Test return: {test_results.get('total_profit', 0.0):.2f}")
        
        return fold_results
        
    def _run_walk_forward_analysis(self, data, component_name):
        """
        Run walk-forward analysis to evaluate strategy performance across time
        
        Args:
            data (pd.DataFrame): Price data
            component_name (str): Component to test
            
        Returns:
            dict: Walk-forward analysis results
        """
        logger.info(f"Running walk-forward analysis for {component_name}")
        
        # Initialize results
        wfa_results = {
            "windows": [],
            "summary": {
                "mean_return": 0.0,
                "mean_sharpe": 0.0,
                "consistency_score": 0.0,
                "regime_stability": 0.0
            }
        }
        
        # Define window parameters
        train_size = len(data) // 3  # Use 1/3 of data for training
        test_size = len(data) // 6   # Use 1/6 of data for testing
        step_size = len(data) // 12  # Shift window by 1/12 of data
        
        # Ensure minimum sizes
        train_size = max(train_size, 500)
        test_size = max(test_size, 250)
        step_size = max(step_size, 100)
        
        # Adjust if data is insufficient
        if len(data) < train_size + test_size:
            logger.warning("Insufficient data for walk-forward analysis")
            train_size = int(len(data) * 0.7)
            test_size = len(data) - train_size
            step_size = test_size
        
        # Run walk-forward analysis
        window_count = 0
        window_returns = []
        window_sharpes = []
        
        # Calculate number of windows
        num_windows = (len(data) - train_size - test_size) // step_size + 1
        num_windows = min(num_windows, 10)  # Cap at 10 windows to avoid excessive computation
        
        for i in range(num_windows):
            # Calculate window indices
            train_start = i * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            # Ensure we don't exceed data bounds
            if test_end > len(data):
                break
            
            logger.info(f"Processing window {i+1}/{num_windows}")
            logger.info(f"Train: {train_start}-{train_end}, Test: {test_start}-{test_end}")
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Get component parameters
            params = self._get_component_params(component_name)
            
            # Train on training data
            train_results = self._run_backtest_on_data(train_data, params)
            
            # Test on test data
            test_results = self._run_backtest_on_data(test_data, params)
            
            # Store window results
            window_result = {
                "window": i + 1,
                "train_period": {
                    "start": train_start,
                    "end": train_end,
                    "size": len(train_data)
                },
                "test_period": {
                    "start": test_start,
                    "end": test_end,
                    "size": len(test_data)
                },
                "train_metrics": {
                    "total_return": train_results.get("total_profit", 0.0),
                    "max_drawdown": train_results.get("max_drawdown", 0.0),
                    "sharpe_ratio": train_results.get("sharpe_ratio", 0.0),
                    "win_rate": train_results.get("win_rate", 0.0),
                    "profit_factor": train_results.get("profit_factor", 0.0),
                    "total_trades": train_results.get("total_trades", 0)
                },
                "test_metrics": {
                    "total_return": test_results.get("total_profit", 0.0),
                    "max_drawdown": test_results.get("max_drawdown", 0.0),
                    "sharpe_ratio": test_results.get("sharpe_ratio", 0.0),
                    "win_rate": test_results.get("win_rate", 0.0),
                    "profit_factor": test_results.get("profit_factor", 0.0),
                    "total_trades": test_results.get("total_trades", 0)
                }
            }
            
            wfa_results["windows"].append(window_result)
            window_count += 1
            
            # Store metrics for summary
            window_returns.append(test_results.get("total_profit", 0.0))
            window_sharpes.append(test_results.get("sharpe_ratio", 0.0))
            
            logger.info(f"Window {i+1} - Train return: {train_results.get('total_profit', 0.0):.2f}, Test return: {test_results.get('total_profit', 0.0):.2f}")
        
        # Calculate summary statistics
        if window_count > 0:
            # Mean return and Sharpe
            wfa_results["summary"]["mean_return"] = np.mean(window_returns)
            wfa_results["summary"]["mean_sharpe"] = np.mean(window_sharpes)
            
            # Consistency score
            returns_cv = np.std(window_returns) / abs(np.mean(window_returns)) if np.mean(window_returns) != 0 else float('inf')
            wfa_results["summary"]["consistency_score"] = max(0, 100 * (1 - min(returns_cv, 1)))
            
            # Regime stability
            # Calculate correlation between consecutive windows
            stability_scores = []
            for i in range(1, len(wfa_results["windows"])):
                prev_window = wfa_results["windows"][i-1]
                curr_window = wfa_results["windows"][i]
                
                # Compare test metrics
                prev_metrics = prev_window["test_metrics"]
                curr_metrics = curr_window["test_metrics"]
                
                # Simple stability score based on return and Sharpe ratio similarity
                return_ratio = min(prev_metrics["total_return"], curr_metrics["total_return"]) / max(abs(prev_metrics["total_return"]), abs(curr_metrics["total_return"]), 0.01)
                sharpe_ratio = min(prev_metrics["sharpe_ratio"], curr_metrics["sharpe_ratio"]) / max(abs(prev_metrics["sharpe_ratio"]), abs(curr_metrics["sharpe_ratio"]), 0.01)
                
                # Adjust for sign changes
                if prev_metrics["total_return"] * curr_metrics["total_return"] < 0:
                    return_ratio *= 0.5
                if prev_metrics["sharpe_ratio"] * curr_metrics["sharpe_ratio"] < 0:
                    sharpe_ratio *= 0.5
                
                # Calculate window stability score
                window_stability = (return_ratio + sharpe_ratio) / 2
                stability_scores.append(max(0, window_stability))
            
            # Calculate overall regime stability
            wfa_results["summary"]["regime_stability"] = np.mean(stability_scores) * 100 if stability_scores else 0.0
            
            logger.info(f"Walk-forward analysis summary:")
            logger.info(f"Mean return: {wfa_results['summary']['mean_return']:.2f}")
            logger.info(f"Mean Sharpe: {wfa_results['summary']['mean_sharpe']:.2f}")
            logger.info(f"Consistency score: {wfa_results['summary']['consistency_score']:.2f}/100")
            logger.info(f"Regime stability: {wfa_results['summary']['regime_stability']:.2f}/100")
        
        return wfa_results
    
    def _calculate_statistical_significance(self, returns):
        """
        Calculate statistical significance of returns using t-test
        
        Args:
            returns (list): List of returns
            
        Returns:
            dict: Statistical significance results
        """
        if not returns or len(returns) < 2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "confidence_level": 0.0
            }
        
        # Convert to numpy array
        returns_array = np.array(returns)
        
        # Perform one-sample t-test against 0
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(returns_array, 0)
        
        # Calculate confidence level
        confidence_level = (1 - p_value) * 100
        
        # Determine significance
        significant = p_value < 0.05
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "confidence_level": float(confidence_level)
        }
    
    def _get_component_params(self, component_name):
        """
        Get parameters for a specific component
        
        Args:
            component_name (str): Component name
            
        Returns:
            dict: Component parameters
        """
        # Default parameters
        default_params = {
            "max_position_size": self.max_position_size,
            "risk_per_trade": 0.02,
            "use_stop_loss": True,
            "use_take_profit": True
        }
        
        # Component-specific parameters
        component_params = {
            "full_ensemble": {
                'use_markov_chain': True,
                'use_ml_assessment': True,
                'use_rnn_selection': True,
                'use_microstructure': True,
                'use_fractal_dimension': True,
                'use_hurst_exponent': True,
                'use_volume_imbalance': True
            },
            "markov_chain": {
                "use_markov_chain": True,
                "markov_order": 2,
                "min_probability": 0.65
            },
            "ml_assessment": {
                "use_ml_assessment": True,
                "confidence_threshold": 0.75,
                "use_random_forest": True
            },
            "rnn_selection": {
                "use_rnn_selection": True,
                "min_confidence": 0.7,
                "use_pseudo_forest": True
            },
            "microstructure": {
                "use_microstructure": True,
                "noise_threshold": 0.3
            },
            "fractal_dimension": {
                "use_fractal_dimension": True,
                "dimension_threshold": 1.5
            },
            "hurst_exponent": {
                "use_hurst_exponent": True,
                "min_exponent": 0.4,
                "max_exponent": 0.6
            },
            "volume_imbalance": {
                "use_volume_imbalance": True,
                "min_imbalance": 1.5
            }
        }
        
        # Get parameters for the component
        params = default_params.copy()
        if component_name in component_params:
            params.update(component_params[component_name])
        
        return params
    
    def _run_backtest_on_data(self, data, params):
        """
        Run backtest on data with specified parameters
        
        Args:
            data (pd.DataFrame): Price data
            params (dict): Backtest parameters
            
        Returns:
            dict: Backtest results
        """
        try:
            # Create backtest instance
            backtest = UltraOptimizedBacktest(
                symbol="CUSTOM",
                timeframe="CUSTOM",
                initial_balance=self.initial_balance,
                max_position_size=params.get("max_position_size", self.max_position_size)
            )
            
            # Set parameters
            backtest.set_parameters(params)
            
            # Run backtest
            results = backtest.run_backtest_on_data(data)
            
            return results
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                "total_profit": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }

    def analyze_profit_factor(self, period_name="1_year", component_name="full_ensemble"):
        """
        Perform detailed analysis of profit factor across different market conditions
        
        Args:
            period_name (str): Period to analyze
            component_name (str): Component to analyze
            
        Returns:
            dict: Profit factor analysis results
        """
        logger.info(f"Analyzing profit factor for period: {period_name}, component: {component_name}")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize analysis results
        pf_analysis = {
            "overall_profit_factor": 0.0,
            "monthly_profit_factors": {},
            "market_condition_profit_factors": {
                "bullish": 0.0,
                "bearish": 0.0,
                "sideways": 0.0,
                "volatile": 0.0,
                "low_volatility": 0.0
            },
            "time_of_day_profit_factors": {},
            "trade_duration_profit_factors": {},
            "position_size_profit_factors": {},
            "consecutive_trades": {
                "after_win": 0.0,
                "after_loss": 0.0
            },
            "optimization_suggestions": []
        }
        
        # Get data for the period
        symbol = self.symbols[0]  # Use first symbol for simplicity
        
        try:
            # Create backtest instance
            backtest = UltraOptimizedBacktest(
                symbol=symbol,
                timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                initial_balance=self.initial_balance,
                max_position_size=self.max_position_size
            )
            
            # Load data for the period
            data = backtest.load_data(
                start_date=period_info["start_date"],
                end_date=period_info["end_date"]
            )
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for period {period_name}")
                return pf_analysis
            
            # Get component parameters
            params = self._get_component_params(component_name)
            
            # Set parameters
            backtest.set_parameters(params)
            
            # Run backtest
            results = backtest.run_backtest_on_data(data)
            
            # Get trade history
            trades = results.get("trades", [])
            
            if not trades:
                logger.warning("No trades found in backtest results")
                return pf_analysis
            
            # Calculate overall profit factor
            total_profit = sum(trade["profit"] for trade in trades if trade["profit"] > 0)
            total_loss = abs(sum(trade["profit"] for trade in trades if trade["profit"] < 0))
            
            overall_pf = total_profit / total_loss if total_loss > 0 else float('inf')
            pf_analysis["overall_profit_factor"] = overall_pf
            
            logger.info(f"Overall profit factor: {overall_pf:.2f}")
            
            # Analyze profit factor by month
            import pandas as pd
            from datetime import datetime
            
            # Convert trades to DataFrame for easier analysis
            trade_data = []
            for trade in trades:
                # Convert timestamp to datetime
                entry_time = datetime.fromtimestamp(trade["entry_time"]) if isinstance(trade["entry_time"], (int, float)) else trade["entry_time"]
                exit_time = datetime.fromtimestamp(trade["exit_time"]) if isinstance(trade["exit_time"], (int, float)) else trade["exit_time"]
                
                trade_data.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "profit": trade["profit"],
                    "direction": trade["direction"],
                    "position_size": trade["position_size"] if "position_size" in trade else 0,
                    "duration": (exit_time - entry_time).total_seconds() / 60.0  # Duration in minutes
                })
            
            if not trade_data:
                logger.warning("Could not process trade data for analysis")
                return pf_analysis
                
            trades_df = pd.DataFrame(trade_data)
            
            # Add month column
            trades_df["month"] = trades_df["entry_time"].dt.strftime("%Y-%m")
            
            # Calculate profit factor by month
            monthly_groups = trades_df.groupby("month")
            
            for month, group in monthly_groups:
                month_profit = sum(trade for trade in group["profit"] if trade > 0)
                month_loss = abs(sum(trade for trade in group["profit"] if trade < 0))
                
                month_pf = month_profit / month_loss if month_loss > 0 else float('inf')
                pf_analysis["monthly_profit_factors"][month] = month_pf
                
                logger.info(f"Month {month} profit factor: {month_pf:.2f}")
            
            # Analyze profit factor by trade duration
            # Define duration buckets in minutes
            duration_buckets = {
                "ultra_short": (0, 5),       # 0-5 minutes
                "short": (5, 60),            # 5-60 minutes
                "medium": (60, 240),         # 1-4 hours
                "long": (240, 1440),         # 4-24 hours
                "very_long": (1440, float('inf'))  # >24 hours
            }
            
            for bucket_name, (min_duration, max_duration) in duration_buckets.items():
                bucket_trades = trades_df[(trades_df["duration"] >= min_duration) & (trades_df["duration"] < max_duration)]
                
                if len(bucket_trades) > 0:
                    bucket_profit = sum(trade for trade in bucket_trades["profit"] if trade > 0)
                    bucket_loss = abs(sum(trade for trade in bucket_trades["profit"] if trade < 0))
                    
                    bucket_pf = bucket_profit / bucket_loss if bucket_loss > 0 else float('inf')
                    pf_analysis["trade_duration_profit_factors"][bucket_name] = bucket_pf
                    
                    logger.info(f"Duration {bucket_name} profit factor: {bucket_pf:.2f}")
            
            # Analyze profit factor by position size
            # Normalize position sizes and create buckets
            if "position_size" in trades_df.columns:
                max_pos_size = trades_df["position_size"].max()
                if max_pos_size > 0:
                    trades_df["position_size_pct"] = trades_df["position_size"] / max_pos_size * 100
                    
                    # Define position size buckets
                    size_buckets = {
                        "very_small": (0, 20),
                        "small": (20, 40),
                        "medium": (40, 60),
                        "large": (60, 80),
                        "very_large": (80, 100)
                    }
                    
                    for bucket_name, (min_size, max_size) in size_buckets.items():
                        bucket_trades = trades_df[(trades_df["position_size_pct"] >= min_size) & (trades_df["position_size_pct"] < max_size)]
                        
                        if len(bucket_trades) > 0:
                            bucket_profit = sum(trade for trade in bucket_trades["profit"] if trade > 0)
                            bucket_loss = abs(sum(trade for trade in bucket_trades["profit"] if trade < 0))
                            
                            bucket_pf = bucket_profit / bucket_loss if bucket_loss > 0 else float('inf')
                            pf_analysis["position_size_profit_factors"][bucket_name] = bucket_pf
                            
                            logger.info(f"Position size {bucket_name} profit factor: {bucket_pf:.2f}")
            
            # Analyze profit factor after consecutive wins/losses
            # First, mark each trade as win or loss
            trades_df["is_win"] = trades_df["profit"] > 0
            
            # Then analyze trades after wins and after losses
            after_win_trades = []
            after_loss_trades = []
            
            for i in range(1, len(trades_df)):
                if trades_df.iloc[i-1]["is_win"]:
                    after_win_trades.append(trades_df.iloc[i]["profit"])
                else:
                    after_loss_trades.append(trades_df.iloc[i]["profit"])
            
            # Calculate profit factor after wins
            if after_win_trades:
                win_profit = sum(trade for trade in after_win_trades if trade > 0)
                win_loss = abs(sum(trade for trade in after_win_trades if trade < 0))
                
                after_win_pf = win_profit / win_loss if win_loss > 0 else float('inf')
                pf_analysis["consecutive_trades"]["after_win"] = after_win_pf
                
                logger.info(f"Profit factor after win: {after_win_pf:.2f}")
            
            # Calculate profit factor after losses
            if after_loss_trades:
                loss_profit = sum(trade for trade in after_loss_trades if trade > 0)
                loss_loss = abs(sum(trade for trade in after_loss_trades if trade < 0))
                
                after_loss_pf = loss_profit / loss_loss if loss_loss > 0 else float('inf')
                pf_analysis["consecutive_trades"]["after_loss"] = after_loss_pf
                
                logger.info(f"Profit factor after loss: {after_loss_pf:.2f}")
            
            # Generate optimization suggestions
            self._generate_pf_optimization_suggestions(pf_analysis)
            
            return pf_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing profit factor: {e}")
            return pf_analysis
    
    def _generate_pf_optimization_suggestions(self, pf_analysis):
        """
        Generate suggestions to optimize profit factor based on analysis
        
        Args:
            pf_analysis (dict): Profit factor analysis results
        """
        suggestions = []
        
        # Check monthly profit factors
        if pf_analysis["monthly_profit_factors"]:
            best_month = max(pf_analysis["monthly_profit_factors"].items(), key=lambda x: x[1])
            worst_month = min(pf_analysis["monthly_profit_factors"].items(), key=lambda x: x[1])
            
            if worst_month[1] < 1.0:
                suggestions.append(f"Consider avoiding trading in months similar to {worst_month[0]} (PF: {worst_month[1]:.2f})")
            
            suggestions.append(f"Optimize for conditions similar to {best_month[0]} (PF: {best_month[1]:.2f})")
        
        # Check trade duration profit factors
        if pf_analysis["trade_duration_profit_factors"]:
            best_duration = max(pf_analysis["trade_duration_profit_factors"].items(), key=lambda x: x[1])
            worst_duration = min(pf_analysis["trade_duration_profit_factors"].items(), key=lambda x: x[1])
            
            if best_duration[1] > 2.0:
                suggestions.append(f"Focus on {best_duration[0]} duration trades (PF: {best_duration[1]:.2f})")
            
            if worst_duration[1] < 1.0:
                suggestions.append(f"Avoid {worst_duration[0]} duration trades (PF: {worst_duration[1]:.2f})")
        
        # Check position size profit factors
        if pf_analysis["position_size_profit_factors"]:
            best_size = max(pf_analysis["position_size_profit_factors"].items(), key=lambda x: x[1])
            
            suggestions.append(f"Optimize position sizing to favor {best_size[0]} positions (PF: {best_size[1]:.2f})")
        
        # Check consecutive trade profit factors
        if pf_analysis["consecutive_trades"]["after_win"] > pf_analysis["consecutive_trades"]["after_loss"]:
            suggestions.append("Consider increasing position size after winning trades")
        else:
            suggestions.append("Consider increasing position size after losing trades (mean reversion)")
        
        # Add suggestions to analysis
        pf_analysis["optimization_suggestions"] = suggestions
        
    def optimize_for_profit_factor(self, period_name="1_year", component_name="full_ensemble", target_pf=50.0):
        """
        Optimize strategy parameters specifically for high profit factor
        
        Args:
            period_name (str): Period to optimize for
            component_name (str): Component to optimize
            target_pf (float): Target profit factor to aim for
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Optimizing {component_name} for high profit factor (target: {target_pf})")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize optimization results
        opt_results = {
            "initial_profit_factor": 0.0,
            "optimized_profit_factor": 0.0,
            "win_rate": 0.0,
            "equity_curve_positive": False,
            "parameter_changes": {},
            "optimization_path": [],
            "trade_count": 0,
            "success": False
        }
        
        # Get data for the period
        symbol = self.symbols[0]  # Use first symbol for simplicity
        
        try:
            # Create backtest instance
            backtest = UltraOptimizedBacktest(
                symbol=symbol,
                timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                initial_balance=self.initial_balance,
                max_position_size=self.max_position_size
            )
            
            # Load data for the period
            data = backtest.load_data(
                start_date=period_info["start_date"],
                end_date=period_info["end_date"]
            )
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for period {period_name}")
                return opt_results
            
            # Get initial parameters
            initial_params = self._get_component_params(component_name)
            
            # Run initial backtest
            backtest.set_parameters(initial_params)
            initial_results = backtest.run_backtest_on_data(data)
            
            # Calculate initial profit factor
            initial_trades = initial_results.get("trades", [])
            if not initial_trades:
                logger.warning("No trades found in initial backtest")
                return opt_results
            
            initial_profit = sum(trade["profit"] for trade in initial_trades if trade["profit"] > 0)
            initial_loss = abs(sum(trade["profit"] for trade in initial_trades if trade["profit"] < 0))
            initial_pf = initial_profit / initial_loss if initial_loss > 0 else float('inf')
            
            # Store initial profit factor
            opt_results["initial_profit_factor"] = initial_pf
            logger.info(f"Initial profit factor: {initial_pf:.2f}")
            
            # Define parameters to optimize
            param_ranges = self._get_optimization_ranges(component_name)
            
            # Initialize best parameters and results
            best_params = initial_params.copy()
            best_pf = initial_pf
            best_win_rate = sum(1 for trade in initial_trades if trade["profit"] > 0) / len(initial_trades) * 100
            best_equity_curve = initial_results.get("equity_curve", [])
            best_trades = initial_trades
            
            # Store initial optimization step
            opt_results["optimization_path"].append({
                "step": 0,
                "profit_factor": initial_pf,
                "win_rate": best_win_rate,
                "params": initial_params.copy()
            })
            
            # Perform optimization
            import itertools
            import random
            
            # First, try grid search for key parameters
            logger.info("Performing grid search optimization")
            
            # Select a subset of parameters to optimize via grid search
            grid_params = {}
            for param, (min_val, max_val, step) in param_ranges.items():
                if param in ["risk_per_trade", "min_probability", "confidence_threshold", "min_confidence"]:
                    values = [min_val + i * step for i in range(int((max_val - min_val) / step) + 1)]
                    grid_params[param] = values
            
            # Generate parameter combinations
            param_keys = list(grid_params.keys())
            param_values = list(grid_params.values())
            
            # Limit to a reasonable number of combinations
            max_combinations = 50
            combinations = list(itertools.product(*param_values))
            if len(combinations) > max_combinations:
                combinations = random.sample(combinations, max_combinations)
            
            # Test each combination
            for i, combination in enumerate(combinations):
                # Create parameter set
                test_params = best_params.copy()
                for j, param in enumerate(param_keys):
                    test_params[param] = combination[j]
                
                # Run backtest with these parameters
                backtest.set_parameters(test_params)
                test_results = backtest.run_backtest_on_data(data)
                
                # Calculate profit factor
                test_trades = test_results.get("trades", [])
                if test_trades and len(test_trades) >= 10:
                    test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                    test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                    test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                    
                    # Calculate win rate
                    test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                    
                    # Check if equity curve is positive
                    test_equity_curve = test_results.get("equity_curve", [])
                    equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                    
                    # Check if this is better
                    if test_pf > best_pf and test_win_rate >= 55.0 and equity_positive:
                        best_params = test_params.copy()
                        best_pf = test_pf
                        best_win_rate = test_win_rate
                        best_equity_curve = test_equity_curve
                        best_trades = test_trades
                        
                        logger.info(f"New best PF: {best_pf:.2f}, Win rate: {best_win_rate:.2f}%")
                
                # Store optimization step
                opt_results["optimization_path"].append({
                    "step": i + 1,
                    "profit_factor": test_pf,
                    "win_rate": test_win_rate,
                    "params": test_params.copy()
                })
            
            # Now try hill climbing to fine-tune
            logger.info("Performing hill climbing optimization")
            
            # Start from the best parameters found so far
            current_params = best_params.copy()
            current_win_rate = best_win_rate
            improvement_found = True
            max_iterations = 20
            iteration = 0
            
            while improvement_found and iteration < max_iterations:
                improvement_found = False
                iteration += 1
                
                # Try adjusting each parameter
                for param, (min_val, max_val, step) in param_ranges.items():
                    # Try increasing the parameter
                    if current_params.get(param, 0) + step <= max_val:
                        test_params = current_params.copy()
                        test_params[param] = current_params.get(param, 0) + step
                        
                        # Run backtest
                        backtest.set_parameters(test_params)
                        test_results = backtest.run_backtest_on_data(data)
                        
                        # Calculate profit factor
                        test_trades = test_results.get("trades", [])
                        if test_trades and len(test_trades) >= 10:
                            # Calculate win rate
                            test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                            
                            # Calculate profit factor
                            test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                            test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                            test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                            
                            # Check if equity curve is positive
                            test_equity_curve = test_results.get("equity_curve", [])
                            equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                            
                            # Check if this is better
                            if test_pf > current_win_rate and test_win_rate >= 55.0 and equity_positive:
                                current_params = test_params.copy()
                                current_win_rate = test_win_rate
                                improvement_found = True
                                
                                # Check if this is the best overall
                                if test_win_rate > best_win_rate:
                                    best_params = test_params.copy()
                                    best_win_rate = test_win_rate
                                    best_equity_curve = test_equity_curve
                                    best_trades = test_trades
                                    
                                    logger.info(f"New best win rate: {best_win_rate:.2f}%")
                    
                    # Try decreasing the parameter
                    if current_params.get(param, 0) - step >= min_val:
                        test_params = current_params.copy()
                        test_params[param] = current_params.get(param, 0) - step
                        
                        # Run backtest
                        backtest.set_parameters(test_params)
                        test_results = backtest.run_backtest_on_data(data)
                        
                        # Calculate profit factor
                        test_trades = test_results.get("trades", [])
                        if test_trades and len(test_trades) >= 10:
                            # Calculate win rate
                            test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                            
                            # Calculate profit factor
                            test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                            test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                            test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                            
                            # Check if equity curve is positive
                            test_equity_curve = test_results.get("equity_curve", [])
                            equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                            
                            # Check if this is better
                            if test_pf > current_win_rate and test_win_rate >= 55.0 and equity_positive:
                                current_params = test_params.copy()
                                current_win_rate = test_win_rate
                                improvement_found = True
                                
                                # Check if this is the best overall
                                if test_win_rate > best_win_rate:
                                    best_params = test_params.copy()
                                    best_win_rate = test_win_rate
                                    best_equity_curve = test_equity_curve
                                    best_trades = test_trades
                                    
                                    logger.info(f"New best win rate: {best_win_rate:.2f}%")
                
                # Store optimization step
                opt_results["optimization_path"].append({
                    "step": len(opt_results["optimization_path"]) + 1,
                    "win_rate": current_win_rate,
                    "profit_factor": best_pf,
                    "params": current_params.copy()
                })
            
            # Store final results
            opt_results["optimized_win_rate"] = best_win_rate
            opt_results["profit_factor"] = best_pf
            opt_results["equity_curve_positive"] = len(best_equity_curve) > 0 and best_equity_curve[-1] > best_equity_curve[0]
            opt_results["trade_count"] = len(best_trades)
            
            # Calculate parameter changes
            for param, value in best_params.items():
                if param in initial_params and value != initial_params[param]:
                    opt_results["parameter_changes"][param] = {
                        "from": initial_params[param],
                        "to": value
                    }
            
            # Determine if optimization was successful
            opt_results["success"] = (best_win_rate >= 65.0 and 
                                     best_pf >= 10.0 and 
                                     opt_results["equity_curve_positive"])
            
            logger.info(f"Optimization completed:")
            logger.info(f"Initial win rate: {initial_win_rate:.2f}% -> Optimized win rate: {best_win_rate:.2f}%")
            logger.info(f"Profit factor: {best_pf:.2f}")
            logger.info(f"Equity curve positive: {opt_results['equity_curve_positive']}")
            logger.info(f"Target achieved: {opt_results['success']}")
            
            return opt_results
            
        except Exception as e:
            logger.error(f"Error optimizing for profit factor: {e}")
            return opt_results
    
    def _get_optimization_ranges(self, component_name):
        """
        Get parameter ranges for optimization
        
        Args:
            component_name (str): Component name
            
        Returns:
            dict: Parameter ranges (min, max, step)
        """
        # Common parameters
        common_ranges = {
            "risk_per_trade": (0.01, 0.05, 0.005),
            "max_position_size": (self.max_position_size * 0.5, self.max_position_size, self.max_position_size * 0.1)
        }
        
        # Component-specific parameters
        component_ranges = {
            "full_ensemble": {
                # No specific parameters for full ensemble
            },
            "markov_chain": {
                "markov_order": (1, 3, 1),
                "min_probability": (0.55, 0.85, 0.05)
            },
            "ml_assessment": {
                "confidence_threshold": (0.6, 0.9, 0.05)
            },
            "rnn_selection": {
                "min_confidence": (0.6, 0.9, 0.05)
            },
            "microstructure": {
                "noise_threshold": (0.1, 0.5, 0.05)
            },
            "fractal_dimension": {
                "dimension_threshold": (1.3, 1.7, 0.05)
            },
            "hurst_exponent": {
                "min_exponent": (0.3, 0.5, 0.05),
                "max_exponent": (0.5, 0.7, 0.05)
            },
            "volume_imbalance": {
                "min_imbalance": (1.2, 2.0, 0.1)
            }
        }
        
        # Combine common and component-specific ranges
        ranges = common_ranges.copy()
        if component_name in component_ranges:
            ranges.update(component_ranges[component_name])
        
        return ranges

    def optimize_for_win_rate(self, period_name="1_year", component_name="full_ensemble", min_pf=10.0, target_win_rate=65.0):
        """
        Optimize strategy parameters for high win rate while maintaining profit factor
        
        Args:
            period_name (str): Period to optimize for
            component_name (str): Component to optimize
            min_pf (float): Minimum acceptable profit factor
            target_win_rate (float): Target win rate percentage
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Optimizing {component_name} for high win rate (target: {target_win_rate}%) while maintaining PF >= {min_pf}")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize optimization results
        opt_results = {
            "initial_win_rate": 0.0,
            "optimized_win_rate": 0.0,
            "profit_factor": 0.0,
            "equity_curve_positive": False,
            "parameter_changes": {},
            "optimization_path": [],
            "trade_count": 0,
            "success": False
        }
        
        # Get data for the period
        symbol = self.symbols[0]  # Use first symbol for simplicity
        
        try:
            # Create backtest instance
            backtest = UltraOptimizedBacktest(
                symbol=symbol,
                timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                initial_balance=self.initial_balance,
                max_position_size=self.max_position_size
            )
            
            # Load data for the period
            data = backtest.load_data(
                start_date=period_info["start_date"],
                end_date=period_info["end_date"]
            )
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for period {period_name}")
                return opt_results
            
            # Get initial parameters
            initial_params = self._get_component_params(component_name)
            
            # Run initial backtest
            backtest.set_parameters(initial_params)
            initial_results = backtest.run_backtest_on_data(data)
            
            # Calculate initial metrics
            initial_trades = initial_results.get("trades", [])
            if not initial_trades:
                logger.warning("No trades found in initial backtest")
                return opt_results
            
            # Calculate win rate
            initial_win_rate = sum(1 for trade in initial_trades if trade["profit"] > 0) / len(initial_trades) * 100
            
            # Calculate profit factor
            initial_profit = sum(trade["profit"] for trade in initial_trades if trade["profit"] > 0)
            initial_loss = abs(sum(trade["profit"] for trade in initial_trades if trade["profit"] < 0))
            initial_pf = initial_profit / initial_loss if initial_loss > 0 else float('inf')
            
            # Store initial metrics
            opt_results["initial_win_rate"] = initial_win_rate
            logger.info(f"Initial win rate: {initial_win_rate:.2f}%, Profit factor: {initial_pf:.2f}")
            
            # Define parameters to optimize
            param_ranges = self._get_optimization_ranges(component_name)
            
            # Initialize best parameters and results
            best_params = initial_params.copy()
            best_win_rate = initial_win_rate
            best_pf = initial_pf
            best_equity_curve = initial_results.get("equity_curve", [])
            best_trades = initial_trades
            
            # Store initial optimization step
            opt_results["optimization_path"].append({
                "step": 0,
                "win_rate": initial_win_rate,
                "profit_factor": initial_pf,
                "params": initial_params.copy()
            })
            
            # Perform optimization
            import itertools
            import random
            
            # First, try grid search for key parameters
            logger.info("Performing grid search optimization")
            
            # Select a subset of parameters to optimize via grid search
            grid_params = {}
            for param, (min_val, max_val, step) in param_ranges.items():
                if param in ["risk_per_trade", "min_probability", "confidence_threshold", "min_confidence"]:
                    values = [min_val + i * step for i in range(int((max_val - min_val) / step) + 1)]
                    grid_params[param] = values
            
            # Generate parameter combinations
            param_keys = list(grid_params.keys())
            param_values = list(grid_params.values())
            
            # Limit to a reasonable number of combinations
            max_combinations = 50
            combinations = list(itertools.product(*param_values))
            if len(combinations) > max_combinations:
                combinations = random.sample(combinations, max_combinations)
            
            # Test each combination
            for i, combination in enumerate(combinations):
                # Create parameter set
                test_params = best_params.copy()
                for j, param in enumerate(param_keys):
                    test_params[param] = combination[j]
                
                # Run backtest with these parameters
                backtest.set_parameters(test_params)
                test_results = backtest.run_backtest_on_data(data)
                
                # Calculate metrics
                test_trades = test_results.get("trades", [])
                if test_trades and len(test_trades) >= 10:
                    # Calculate win rate
                    test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                    
                    # Calculate profit factor
                    test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                    test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                    test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                    
                    # Check if equity curve is positive
                    test_equity_curve = test_results.get("equity_curve", [])
                    equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                    
                    # Check if this is better
                    if test_win_rate > best_win_rate and test_pf >= min_pf and equity_positive:
                        best_params = test_params.copy()
                        best_win_rate = test_win_rate
                        best_pf = test_pf
                        best_equity_curve = test_equity_curve
                        best_trades = test_trades
                        
                        logger.info(f"New best win rate: {best_win_rate:.2f}%, PF: {best_pf:.2f}")
                
                # Store optimization step
                opt_results["optimization_path"].append({
                    "step": i + 1,
                    "win_rate": test_win_rate,
                    "profit_factor": test_pf,
                    "params": test_params.copy()
                })
            
            # Now try hill climbing to fine-tune
            logger.info("Performing hill climbing optimization")
            
            # Start from the best parameters found so far
            current_params = best_params.copy()
            current_win_rate = best_win_rate
            improvement_found = True
            max_iterations = 20
            iteration = 0
            
            while improvement_found and iteration < max_iterations:
                improvement_found = False
                iteration += 1
                
                # Try adjusting each parameter
                for param, (min_val, max_val, step) in param_ranges.items():
                    # Try increasing the parameter
                    if current_params.get(param, 0) + step <= max_val:
                        test_params = current_params.copy()
                        test_params[param] = current_params.get(param, 0) + step
                        
                        # Run backtest
                        backtest.set_parameters(test_params)
                        test_results = backtest.run_backtest_on_data(data)
                        
                        # Calculate metrics
                        test_trades = test_results.get("trades", [])
                        if test_trades and len(test_trades) >= 10:
                            # Calculate win rate
                            test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                            
                            # Calculate profit factor
                            test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                            test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                            test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                            
                            # Check if equity curve is positive
                            test_equity_curve = test_results.get("equity_curve", [])
                            equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                            
                            # Check if this is better
                            if test_win_rate > current_win_rate and test_pf >= min_pf and equity_positive:
                                current_params = test_params.copy()
                                current_win_rate = test_win_rate
                                improvement_found = True
                                
                                # Check if this is the best overall
                                if test_win_rate > best_win_rate:
                                    best_params = test_params.copy()
                                    best_win_rate = test_win_rate
                                    best_pf = test_pf
                                    best_equity_curve = test_equity_curve
                                    best_trades = test_trades
                                    
                                    logger.info(f"New best win rate: {best_win_rate:.2f}%, PF: {best_pf:.2f}")
                    
                    # Try decreasing the parameter
                    if current_params.get(param, 0) - step >= min_val:
                        test_params = current_params.copy()
                        test_params[param] = current_params.get(param, 0) - step
                        
                        # Run backtest
                        backtest.set_parameters(test_params)
                        test_results = backtest.run_backtest_on_data(data)
                        
                        # Calculate metrics
                        test_trades = test_results.get("trades", [])
                        if test_trades and len(test_trades) >= 10:
                            # Calculate win rate
                            test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                            
                            # Calculate profit factor
                            test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                            test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                            test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                            
                            # Check if equity curve is positive
                            test_equity_curve = test_results.get("equity_curve", [])
                            equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                            
                            # Check if this is better
                            if test_win_rate > current_win_rate and test_pf >= min_pf and equity_positive:
                                current_params = test_params.copy()
                                current_win_rate = test_win_rate
                                improvement_found = True
                                
                                # Check if this is the best overall
                                if test_win_rate > best_win_rate:
                                    best_params = test_params.copy()
                                    best_win_rate = test_win_rate
                                    best_pf = test_pf
                                    best_equity_curve = test_equity_curve
                                    best_trades = test_trades
                                    
                                    logger.info(f"New best win rate: {best_win_rate:.2f}%, PF: {best_pf:.2f}")
                
                # Store optimization step
                opt_results["optimization_path"].append({
                    "step": len(opt_results["optimization_path"]) + 1,
                    "win_rate": current_win_rate,
                    "profit_factor": best_pf,
                    "params": current_params.copy()
                })
            
            # Store final results
            opt_results["optimized_win_rate"] = best_win_rate
            opt_results["profit_factor"] = best_pf
            opt_results["equity_curve_positive"] = len(best_equity_curve) > 0 and best_equity_curve[-1] > best_equity_curve[0]
            opt_results["trade_count"] = len(best_trades)
            
            # Calculate parameter changes
            for param, value in best_params.items():
                if param in initial_params and value != initial_params[param]:
                    opt_results["parameter_changes"][param] = {
                        "from": initial_params[param],
                        "to": value
                    }
            
            # Determine if optimization was successful
            opt_results["success"] = (best_win_rate >= target_win_rate and 
                                     best_pf >= min_pf and 
                                     opt_results["equity_curve_positive"])
            
            logger.info(f"Win rate optimization completed:")
            logger.info(f"Initial win rate: {initial_win_rate:.2f}% -> Optimized win rate: {best_win_rate:.2f}%")
            logger.info(f"Profit factor: {best_pf:.2f}")
            logger.info(f"Equity curve positive: {opt_results['equity_curve_positive']}")
            logger.info(f"Target achieved: {opt_results['success']}")
            
            return opt_results
            
        except Exception as e:
            logger.error(f"Error optimizing for win rate: {e}")
            return opt_results
            
    def optimize_combined_metrics(self, period_name="1_year", component_name="full_ensemble", target_pf=50.0, target_win_rate=55.0):
        """
        Optimize strategy parameters for both profit factor and win rate simultaneously
        
        Args:
            period_name (str): Period to optimize for
            component_name (str): Component to optimize
            target_pf (float): Target profit factor
            target_win_rate (float): Target win rate percentage
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Performing combined optimization for {component_name} targeting PF={target_pf} and win rate={target_win_rate}%")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize optimization results
        opt_results = {
            "initial_profit_factor": 0.0,
            "initial_win_rate": 0.0,
            "optimized_profit_factor": 0.0,
            "optimized_win_rate": 0.0,
            "equity_curve_positive": False,
            "parameter_changes": {},
            "optimization_path": [],
            "trade_count": 0,
            "success": False,
            "optimization_score": 0.0
        }
        
        # Get data for the period
        symbol = self.symbols[0]  # Use first symbol for simplicity
        
        try:
            # Create backtest instance
            backtest = UltraOptimizedBacktest(
                symbol=symbol,
                timeframe=self.timeframes[0],  # Use first timeframe for simplicity
                initial_balance=self.initial_balance,
                max_position_size=self.max_position_size
            )
            
            # Load data for the period
            data = backtest.load_data(
                start_date=period_info["start_date"],
                end_date=period_info["end_date"]
            )
            
            if data is None or len(data) == 0:
                logger.error(f"No data available for period {period_name}")
                return opt_results
            
            # Get initial parameters
            initial_params = self._get_component_params(component_name)
            
            # Run initial backtest
            backtest.set_parameters(initial_params)
            initial_results = backtest.run_backtest_on_data(data)
            
            # Calculate initial metrics
            initial_trades = initial_results.get("trades", [])
            if not initial_trades:
                logger.warning("No trades found in initial backtest")
                return opt_results
            
            # Calculate win rate
            initial_win_rate = sum(1 for trade in initial_trades if trade["profit"] > 0) / len(initial_trades) * 100
            
            # Calculate profit factor
            initial_profit = sum(trade["profit"] for trade in initial_trades if trade["profit"] > 0)
            initial_loss = abs(sum(trade["profit"] for trade in initial_trades if trade["profit"] < 0))
            initial_pf = initial_profit / initial_loss if initial_loss > 0 else float('inf')
            
            # Store initial metrics
            opt_results["initial_profit_factor"] = initial_pf
            opt_results["initial_win_rate"] = initial_win_rate
            logger.info(f"Initial metrics - PF: {initial_pf:.2f}, Win rate: {initial_win_rate:.2f}%")
            
            # Define parameters to optimize
            param_ranges = self._get_optimization_ranges(component_name)
            
            # Initialize best parameters and results
            best_params = initial_params.copy()
            best_pf = initial_pf
            best_win_rate = initial_win_rate
            best_score = self._calculate_optimization_score(initial_pf, initial_win_rate, target_pf, target_win_rate)
            best_equity_curve = initial_results.get("equity_curve", [])
            best_trades = initial_trades
            
            # Store initial optimization step
            opt_results["optimization_path"].append({
                "step": 0,
                "profit_factor": initial_pf,
                "win_rate": initial_win_rate,
                "score": best_score,
                "params": initial_params.copy()
            })
            
            # Perform optimization
            import itertools
            import random
            
            # First, try grid search for key parameters
            logger.info("Performing grid search optimization")
            
            # Select a subset of parameters to optimize via grid search
            grid_params = {}
            for param, (min_val, max_val, step) in param_ranges.items():
                if param in ["risk_per_trade", "min_probability", "confidence_threshold", "min_confidence", 
                           "stop_loss_atr_multiplier", "take_profit_atr_multiplier"]:
                    values = [min_val + i * step for i in range(int((max_val - min_val) / step) + 1)]
                    grid_params[param] = values
            
            # Generate parameter combinations
            param_keys = list(grid_params.keys())
            param_values = list(grid_params.values())
            
            # Limit to a reasonable number of combinations
            max_combinations = 100
            combinations = list(itertools.product(*param_values))
            if len(combinations) > max_combinations:
                combinations = random.sample(combinations, max_combinations)
            
            # Test each combination
            for i, combination in enumerate(combinations):
                # Create parameter set
                test_params = best_params.copy()
                for j, param in enumerate(param_keys):
                    test_params[param] = combination[j]
                
                # Run backtest with these parameters
                backtest.set_parameters(test_params)
                test_results = backtest.run_backtest_on_data(data)
                
                # Calculate metrics
                test_trades = test_results.get("trades", [])
                if test_trades and len(test_trades) >= 20:
                    # Calculate win rate
                    test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                    
                    # Calculate profit factor
                    test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                    test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                    test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                    
                    # Check if equity curve is positive
                    test_equity_curve = test_results.get("equity_curve", [])
                    equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                    
                    # Calculate optimization score
                    test_score = self._calculate_optimization_score(test_pf, test_win_rate, target_pf, target_win_rate)
                    
                    # Check if this is better
                    if test_score > best_score and equity_positive:
                        best_params = test_params.copy()
                        best_pf = test_pf
                        best_win_rate = test_win_rate
                        best_score = test_score
                        best_equity_curve = test_equity_curve
                        best_trades = test_trades
                        
                        logger.info(f"New best score: {best_score:.2f} (PF: {best_pf:.2f}, Win rate: {best_win_rate:.2f}%)")
                
                # Store optimization step
                opt_results["optimization_path"].append({
                    "step": i + 1,
                    "profit_factor": test_pf,
                    "win_rate": test_win_rate,
                    "score": test_score,
                    "params": test_params.copy()
                })
            
            # Now try hill climbing to fine-tune
            logger.info("Performing hill climbing optimization")
            
            # Start from the best parameters found so far
            current_params = best_params.copy()
            current_score = best_score
            improvement_found = True
            max_iterations = 30
            iteration = 0
            
            while improvement_found and iteration < max_iterations:
                improvement_found = False
                iteration += 1
                
                # Try adjusting each parameter
                for param, (min_val, max_val, step) in param_ranges.items():
                    # Try increasing the parameter
                    if current_params.get(param, 0) + step <= max_val:
                        test_params = current_params.copy()
                        test_params[param] = current_params.get(param, 0) + step
                        
                        # Run backtest
                        backtest.set_parameters(test_params)
                        test_results = backtest.run_backtest_on_data(data)
                        
                        # Calculate metrics
                        test_trades = test_results.get("trades", [])
                        if test_trades and len(test_trades) >= 20:
                            # Calculate win rate
                            test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                            
                            # Calculate profit factor
                            test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                            test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                            test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                            
                            # Check if equity curve is positive
                            test_equity_curve = test_results.get("equity_curve", [])
                            equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                            
                            # Calculate optimization score
                            test_score = self._calculate_optimization_score(test_pf, test_win_rate, target_pf, target_win_rate)
                            
                            # Check if this is better
                            if test_score > current_score and equity_positive:
                                current_params = test_params.copy()
                                current_score = test_score
                                improvement_found = True
                                
                                # Check if this is the best overall
                                if test_score > best_score:
                                    best_params = test_params.copy()
                                    best_pf = test_pf
                                    best_win_rate = test_win_rate
                                    best_score = test_score
                                    best_equity_curve = test_equity_curve
                                    best_trades = test_trades
                                    
                                    logger.info(f"New best score: {best_score:.2f} (PF: {best_pf:.2f}, Win rate: {best_win_rate:.2f}%)")
                    
                    # Try decreasing the parameter
                    if current_params.get(param, 0) - step >= min_val:
                        test_params = current_params.copy()
                        test_params[param] = current_params.get(param, 0) - step
                        
                        # Run backtest
                        backtest.set_parameters(test_params)
                        test_results = backtest.run_backtest_on_data(data)
                        
                        # Calculate metrics
                        test_trades = test_results.get("trades", [])
                        if test_trades and len(test_trades) >= 20:
                            # Calculate win rate
                            test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                            
                            # Calculate profit factor
                            test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                            test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                            test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                            
                            # Check if equity curve is positive
                            test_equity_curve = test_results.get("equity_curve", [])
                            equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                            
                            # Calculate optimization score
                            test_score = self._calculate_optimization_score(test_pf, test_win_rate, target_pf, target_win_rate)
                            
                            # Check if this is better
                            if test_score > current_score and equity_positive:
                                current_params = test_params.copy()
                                current_score = test_score
                                improvement_found = True
                                
                                # Check if this is the best overall
                                if test_score > best_score:
                                    best_params = test_params.copy()
                                    best_pf = test_pf
                                    best_win_rate = test_win_rate
                                    best_score = test_score
                                    best_equity_curve = test_equity_curve
                                    best_trades = test_trades
                                    
                                    logger.info(f"New best score: {best_score:.2f} (PF: {best_pf:.2f}, Win rate: {best_win_rate:.2f}%)")
                
                # Store optimization step
                opt_results["optimization_path"].append({
                    "step": len(opt_results["optimization_path"]) + 1,
                    "profit_factor": best_pf,
                    "win_rate": best_win_rate,
                    "score": current_score,
                    "params": current_params.copy()
                })
            
            # Store final results
            opt_results["optimized_profit_factor"] = best_pf
            opt_results["optimized_win_rate"] = best_win_rate
            opt_results["optimization_score"] = best_score
            opt_results["equity_curve_positive"] = len(best_equity_curve) > 0 and best_equity_curve[-1] > best_equity_curve[0]
            opt_results["trade_count"] = len(best_trades)
            
            # Calculate parameter changes
            for param, value in best_params.items():
                if param in initial_params and value != initial_params[param]:
                    opt_results["parameter_changes"][param] = {
                        "from": initial_params[param],
                        "to": value
                    }
            
            # Determine if optimization was successful
            opt_results["success"] = (best_pf >= target_pf and 
                                     best_win_rate >= target_win_rate and 
                                     opt_results["equity_curve_positive"])
            
            logger.info(f"Combined optimization completed:")
            logger.info(f"Initial PF: {initial_pf:.2f} -> Optimized PF: {best_pf:.2f}")
            logger.info(f"Initial win rate: {initial_win_rate:.2f}% -> Optimized win rate: {best_win_rate:.2f}%")
            logger.info(f"Equity curve positive: {opt_results['equity_curve_positive']}")
            logger.info(f"Target achieved: {opt_results['success']}")
            
            # If successful, update component parameters
            if opt_results["success"] and self.auto_update_params:
                self._update_component_params(component_name, best_params)
                logger.info(f"Parameters for {component_name} have been automatically updated")
            
            return opt_results
            
        except Exception as e:
            logger.error(f"Error in combined optimization: {e}")
            return opt_results
            
    def _calculate_optimization_score(self, profit_factor, win_rate, target_pf, target_win_rate):
        """
        Calculate a combined optimization score based on profit factor and win rate
        
        Args:
            profit_factor (float): Current profit factor
            win_rate (float): Current win rate
            target_pf (float): Target profit factor
            target_win_rate (float): Target win rate
            
        Returns:
            float: Combined score (higher is better)
        """
        # Calculate how close we are to targets (or how much we exceed them)
        pf_score = min(profit_factor / target_pf, 2.0)  # Cap at 2x target
        wr_score = min(win_rate / target_win_rate, 1.5)  # Cap at 1.5x target
        
        # Weight profit factor more heavily (70/30 split)
        combined_score = (0.7 * pf_score) + (0.3 * wr_score)
        
        # Apply penalties if below targets
        if profit_factor < target_pf:
            combined_score *= 0.8
        
        if win_rate < target_win_rate:
            combined_score *= 0.9
            
        return combined_score

    def optimize_multi_timeframe(self, period_name="1_year", component_name="full_ensemble", target_pf=50.0, target_win_rate=55.0):
        """
        Optimize strategy parameters across multiple timeframes to ensure robustness
        
        Args:
            period_name (str): Period to optimize for
            component_name (str): Component to optimize
            target_pf (float): Target profit factor
            target_win_rate (float): Target win rate percentage
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Performing multi-timeframe optimization for {component_name}")
        
        # Get period info
        period_info = self.test_periods.get(period_name)
        if not period_info:
            logger.error(f"Invalid period: {period_name}")
            return {}
        
        # Initialize results
        results = {
            "timeframe_results": {},
            "combined_score": 0.0,
            "best_parameters": {},
            "parameter_stability": {},
            "success": False,
            "summary": {}
        }
        
        # Get initial parameters
        initial_params = self._get_component_params(component_name)
        
        # Track parameter values across timeframes
        param_values = {}
        
        # Optimize for each timeframe
        for timeframe in self.timeframes:
            logger.info(f"Optimizing for timeframe: {timeframe}")
            
            try:
                # Create backtest instance
                backtest = UltraOptimizedBacktest(
                    symbol=self.symbols[0],  # Use first symbol for simplicity
                    timeframe=timeframe,
                    initial_balance=self.initial_balance,
                    max_position_size=self.max_position_size
                )
                
                # Load data for the period
                data = backtest.load_data(
                    start_date=period_info["start_date"],
                    end_date=period_info["end_date"]
                )
                
                if data is None or len(data) == 0:
                    logger.warning(f"No data available for timeframe {timeframe} in period {period_name}")
                    continue
                
                # Run optimization for this timeframe
                # Use a lower target for individual timeframes to allow for averaging
                tf_target_pf = target_pf * 0.7  # 70% of the overall target
                tf_target_wr = target_win_rate * 0.9  # 90% of the overall target
                
                # Run combined optimization
                tf_results = self._run_single_timeframe_optimization(
                    backtest, 
                    data, 
                    component_name, 
                    initial_params,
                    tf_target_pf, 
                    tf_target_wr
                )
                
                # Store results for this timeframe
                results["timeframe_results"][timeframe] = {
                    "profit_factor": tf_results.get("optimized_profit_factor", 0),
                    "win_rate": tf_results.get("optimized_win_rate", 0),
                    "equity_curve_positive": tf_results.get("equity_curve_positive", False),
                    "trade_count": tf_results.get("trade_count", 0),
                    "success": tf_results.get("success", False),
                    "parameters": tf_results.get("best_params", {})
                }
                
                # Track parameter values
                for param, value in tf_results.get("best_params", {}).items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
                
            except Exception as e:
                logger.error(f"Error optimizing for timeframe {timeframe}: {e}")
                continue
        
        # Calculate parameter stability (coefficient of variation for each parameter)
        for param, values in param_values.items():
            if len(values) >= 2:  # Need at least 2 values to calculate stability
                mean = sum(values) / len(values)
                if mean != 0:
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    cv = std_dev / abs(mean)  # Coefficient of variation
                    
                    # Convert to stability score (0-100, higher is better)
                    stability = max(0, 100 * (1 - min(cv, 1)))
                    results["parameter_stability"][param] = stability
        
        # Calculate average parameter values for final recommendation
        best_params = {}
        for param, values in param_values.items():
            if values:
                # For some parameters, we want to be conservative (use min or max)
                if param in ["stop_loss_atr_multiplier"]:
                    # For stop loss, use a slightly larger value for safety
                    best_params[param] = min(values) * 1.1
                elif param in ["take_profit_atr_multiplier"]:
                    # For take profit, use a slightly smaller value for consistency
                    best_params[param] = max(values) * 0.9
                elif param in ["min_probability", "confidence_threshold", "min_confidence"]:
                    # For confidence thresholds, use the maximum (most conservative)
                    best_params[param] = max(values)
                else:
                    # For other parameters, use the average
                    best_params[param] = sum(values) / len(values)
        
        # Store best parameters
        results["best_parameters"] = best_params
        
        # Calculate combined score across timeframes
        tf_scores = []
        for tf, tf_result in results["timeframe_results"].items():
            pf = tf_result["profit_factor"]
            wr = tf_result["win_rate"]
            if pf > 0 and wr > 0:
                score = self._calculate_optimization_score(pf, wr, target_pf, target_win_rate)
                tf_scores.append(score)
        
        if tf_scores:
            results["combined_score"] = sum(tf_scores) / len(tf_scores)
        
        # Calculate summary statistics
        successful_timeframes = sum(1 for tf_result in results["timeframe_results"].values() if tf_result["success"])
        total_timeframes = len(results["timeframe_results"])
        
        # Calculate average metrics
        avg_pf = sum(tf_result["profit_factor"] for tf_result in results["timeframe_results"].values()) / total_timeframes if total_timeframes > 0 else 0
        avg_wr = sum(tf_result["win_rate"] for tf_result in results["timeframe_results"].values()) / total_timeframes if total_timeframes > 0 else 0
        
        # Calculate parameter stability score (average of all parameter stabilities)
        param_stability_scores = list(results["parameter_stability"].values())
        avg_param_stability = sum(param_stability_scores) / len(param_stability_scores) if param_stability_scores else 0
        
        # Store summary
        results["summary"] = {
            "successful_timeframes": successful_timeframes,
            "total_timeframes": total_timeframes,
            "success_rate": (successful_timeframes / total_timeframes * 100) if total_timeframes > 0 else 0,
            "average_profit_factor": avg_pf,
            "average_win_rate": avg_wr,
            "parameter_stability": avg_param_stability
        }
        
        # Determine overall success
        # We consider it successful if at least 70% of timeframes are successful and parameter stability is high
        results["success"] = (
            results["summary"]["success_rate"] >= 70 and 
            avg_param_stability >= 75 and
            avg_pf >= target_pf * 0.9 and  # At least 90% of target PF
            avg_wr >= target_win_rate * 0.95  # At least 95% of target win rate
        )
        
        # Log results
        logger.info(f"Multi-timeframe optimization completed for {component_name}")
        logger.info(f"Success rate: {results['summary']['success_rate']:.2f}% ({successful_timeframes}/{total_timeframes} timeframes)")
        logger.info(f"Average profit factor: {avg_pf:.2f}")
        logger.info(f"Average win rate: {avg_wr:.2f}%")
        logger.info(f"Parameter stability: {avg_param_stability:.2f}/100")
        logger.info(f"Overall success: {results['success']}")
        
        # If successful and auto-update is enabled, update the parameters
        if results["success"] and self.auto_update_params:
            self._update_component_params(component_name, best_params)
            logger.info(f"Parameters for {component_name} have been automatically updated with multi-timeframe optimized values")
        
        return results
    
    def _run_single_timeframe_optimization(self, backtest, data, component_name, initial_params, target_pf, target_win_rate):
        """
        Run optimization for a single timeframe
        
        Args:
            backtest: Backtest instance
            data: Data for the timeframe
            component_name: Component to test
            initial_params: Initial parameters
            target_pf: Target profit factor
            target_win_rate: Target win rate
            
        Returns:
            dict: Optimization results
        """
        # Initialize results
        results = {
            "initial_profit_factor": 0.0,
            "initial_win_rate": 0.0,
            "optimized_profit_factor": 0.0,
            "optimized_win_rate": 0.0,
            "equity_curve_positive": False,
            "best_params": {},
            "trade_count": 0,
            "success": False
        }
        
        try:
            # Run initial backtest
            backtest.set_parameters(initial_params)
            initial_results = backtest.run_backtest_on_data(data)
            
            # Calculate initial metrics
            initial_trades = initial_results.get("trades", [])
            if not initial_trades:
                logger.warning("No trades found in initial backtest")
                return results
            
            # Calculate win rate
            initial_win_rate = sum(1 for trade in initial_trades if trade["profit"] > 0) / len(initial_trades) * 100
            
            # Calculate profit factor
            initial_profit = sum(trade["profit"] for trade in initial_trades if trade["profit"] > 0)
            initial_loss = abs(sum(trade["profit"] for trade in initial_trades if trade["profit"] < 0))
            initial_pf = initial_profit / initial_loss if initial_loss > 0 else float('inf')
            
            # Store initial metrics
            results["initial_profit_factor"] = initial_pf
            results["initial_win_rate"] = initial_win_rate
            
            # Define parameters to optimize
            param_ranges = self._get_optimization_ranges(component_name)
            
            # Initialize best parameters and results
            best_params = initial_params.copy()
            best_pf = initial_pf
            best_win_rate = initial_win_rate
            best_score = self._calculate_optimization_score(initial_pf, initial_win_rate, target_pf, target_win_rate)
            best_equity_curve = initial_results.get("equity_curve", [])
            best_trades = initial_trades
            
            # Perform optimization
            import itertools
            import random
            
            # Select a subset of parameters to optimize via grid search
            grid_params = {}
            for param, (min_val, max_val, step) in param_ranges.items():
                if param in ["risk_per_trade", "min_probability", "confidence_threshold", "min_confidence", 
                           "stop_loss_atr_multiplier", "take_profit_atr_multiplier"]:
                    values = [min_val + i * step for i in range(int((max_val - min_val) / step) + 1)]
                    grid_params[param] = values
            
            # Generate parameter combinations
            param_keys = list(grid_params.keys())
            param_values = list(grid_params.values())
            
            # Limit to a reasonable number of combinations
            max_combinations = 50
            combinations = list(itertools.product(*param_values))
            if len(combinations) > max_combinations:
                combinations = random.sample(combinations, max_combinations)
            
            # Test each combination
            for combination in combinations:
                # Create parameter set
                test_params = best_params.copy()
                for j, param in enumerate(param_keys):
                    test_params[param] = combination[j]
                
                # Run backtest with these parameters
                backtest.set_parameters(test_params)
                test_results = backtest.run_backtest_on_data(data)
                
                # Calculate metrics
                test_trades = test_results.get("trades", [])
                if test_trades and len(test_trades) >= 20:
                    # Calculate win rate
                    test_win_rate = sum(1 for trade in test_trades if trade["profit"] > 0) / len(test_trades) * 100
                    
                    # Calculate profit factor
                    test_profit = sum(trade["profit"] for trade in test_trades if trade["profit"] > 0)
                    test_loss = abs(sum(trade["profit"] for trade in test_trades if trade["profit"] < 0))
                    test_pf = test_profit / test_loss if test_loss > 0 else float('inf')
                    
                    # Check if equity curve is positive
                    test_equity_curve = test_results.get("equity_curve", [])
                    equity_positive = len(test_equity_curve) > 0 and test_equity_curve[-1] > test_equity_curve[0]
                    
                    # Calculate optimization score
                    test_score = self._calculate_optimization_score(test_pf, test_win_rate, target_pf, target_win_rate)
                    
                    # Check if this is better
                    if test_score > best_score and equity_positive:
                        best_params = test_params.copy()
                        best_pf = test_pf
                        best_win_rate = test_win_rate
                        best_score = test_score
                        best_equity_curve = test_equity_curve
                        best_trades = test_trades
            
            # Store final results
            results["optimized_profit_factor"] = best_pf
            results["optimized_win_rate"] = best_win_rate
            results["equity_curve_positive"] = len(best_equity_curve) > 0 and best_equity_curve[-1] > best_equity_curve[0]
            results["trade_count"] = len(best_trades)
            results["best_params"] = best_params
            
            # Determine if optimization was successful
            results["success"] = (best_pf >= target_pf and 
                                 best_win_rate >= target_win_rate and 
                                 results["equity_curve_positive"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in single timeframe optimization: {e}")
            return results

    def save_optimization_results(self, results, optimization_type, component_name, output_dir=None):
        """
        Save optimization results to files
        
        Args:
            results (dict): Optimization results
            optimization_type (str): Type of optimization performed
            component_name (str): Component that was optimized
            output_dir (str): Directory to save results to (default: ./optimization_results)
            
        Returns:
            str: Path to the saved results
        """
        import os
        import json
        import pandas as pd
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "optimization_results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base filename
        base_filename = f"{timestamp}_{optimization_type}_{component_name}"
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Create summary CSV
        if "summary" in results:
            summary_df = pd.DataFrame([results["summary"]])
            summary_path = os.path.join(output_dir, f"{base_filename}_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
        # Create parameters CSV
        if "best_parameters" in results:
            params_df = pd.DataFrame([results["best_parameters"]])
            params_path = os.path.join(output_dir, f"{base_filename}_parameters.csv")
            params_df.to_csv(params_path, index=False)
            
        logger.info(f"Optimization results saved to {json_path}")
        return json_path
    
    def run_high_performance_backtest(self, period_name="1_year", component_names=None, output_dir=None):
        """
        Run a complete high-performance backtest with all optimization methods
        
        Args:
            period_name (str): Period to optimize for
            component_names (list): List of components to optimize, if None, use all components
            output_dir (str): Directory to save results to
            
        Returns:
            dict: Comprehensive results from all optimization methods
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "high_performance_results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get components to optimize
        if component_names is None:
            component_names = ["markov_chain", "ml_assessment", "rnn_strategy", "full_ensemble"]
            
        # Initialize results
        results = {
            "timestamp": timestamp,
            "period_name": period_name,
            "components": {},
            "overall_success": False
        }
        
        # Track success count
        success_count = 0
        
        # Step 1: Optimize each component individually
        for component_name in component_names:
            logger.info(f"Starting high-performance optimization for component: {component_name}")
            
            component_results = {
                "profit_factor_optimization": None,
                "win_rate_optimization": None,
                "combined_metrics_optimization": None,
                "multi_timeframe_optimization": None,
                "success": False
            }
            
            # Step 1.1: Optimize for profit factor
            logger.info(f"Optimizing {component_name} for profit factor")
            pf_results = self.optimize_for_profit_factor(
                period_name=period_name,
                component_name=component_name,
                target_pf=50.0
            )
            
            # Save results
            pf_path = self.save_optimization_results(
                pf_results,
                "profit_factor",
                component_name,
                output_dir=os.path.join(output_dir, component_name)
            )
            
            component_results["profit_factor_optimization"] = {
                "results_path": pf_path,
                "success": pf_results.get("success", False),
                "profit_factor": pf_results.get("optimized_profit_factor", 0),
                "win_rate": pf_results.get("optimized_win_rate", 0)
            }
            
            # Step 1.2: Optimize for win rate
            logger.info(f"Optimizing {component_name} for win rate")
            wr_results = self.optimize_for_win_rate(
                period_name=period_name,
                component_name=component_name,
                target_win_rate=55.0
            )
            
            # Save results
            wr_path = self.save_optimization_results(
                wr_results,
                "win_rate",
                component_name,
                output_dir=os.path.join(output_dir, component_name)
            )
            
            component_results["win_rate_optimization"] = {
                "results_path": wr_path,
                "success": wr_results.get("success", False),
                "profit_factor": wr_results.get("optimized_profit_factor", 0),
                "win_rate": wr_results.get("optimized_win_rate", 0)
            }
            
            # Step 1.3: Optimize for combined metrics
            logger.info(f"Optimizing {component_name} for combined metrics")
            cm_results = self.optimize_combined_metrics(
                period_name=period_name,
                component_name=component_name,
                target_pf=50.0,
                target_win_rate=55.0
            )
            
            # Save results
            cm_path = self.save_optimization_results(
                cm_results,
                "combined_metrics",
                component_name,
                output_dir=os.path.join(output_dir, component_name)
            )
            
            component_results["combined_metrics_optimization"] = {
                "results_path": cm_path,
                "success": cm_results.get("success", False),
                "profit_factor": cm_results.get("optimized_profit_factor", 0),
                "win_rate": cm_results.get("optimized_win_rate", 0),
                "combined_score": cm_results.get("best_score", 0)
            }
            
            # Step 1.4: Optimize across multiple timeframes
            logger.info(f"Optimizing {component_name} across multiple timeframes")
            mt_results = self.optimize_multi_timeframe(
                period_name=period_name,
                component_name=component_name,
                target_pf=50.0,
                target_win_rate=55.0
            )
            
            # Save results
            mt_path = self.save_optimization_results(
                mt_results,
                "multi_timeframe",
                component_name,
                output_dir=os.path.join(output_dir, component_name)
            )
            
            component_results["multi_timeframe_optimization"] = {
                "results_path": mt_path,
                "success": mt_results.get("success", False),
                "combined_score": mt_results.get("combined_score", 0),
                "parameter_stability": mt_results.get("summary", {}).get("parameter_stability", 0)
            }
            
            # Determine overall component success
            if mt_results.get("success", False):
                component_results["success"] = True
                component_results["best_optimization"] = "multi_timeframe"
                success_count += 1
            elif cm_results.get("success", False):
                component_results["success"] = True
                component_results["best_optimization"] = "combined_metrics"
                success_count += 1
            elif pf_results.get("success", False):
                component_results["success"] = True
                component_results["best_optimization"] = "profit_factor"
                success_count += 1
            elif wr_results.get("success", False):
                component_results["success"] = True
                component_results["best_optimization"] = "win_rate"
                success_count += 1
            
            # Store component results
            results["components"][component_name] = component_results
            
        # Step 2: Run cross-validation on the full ensemble
        if "full_ensemble" in component_names:
            logger.info("Running cross-validation on full ensemble")
            cv_results = self.run_cross_validation(
                period_name=period_name,
                component_name="full_ensemble",
                folds=5
            )
            
            # Save cross-validation results
            cv_path = os.path.join(output_dir, f"{timestamp}_cross_validation_full_ensemble.json")
            with open(cv_path, 'w') as f:
                json.dump(cv_results, f, indent=4)
                
            results["cross_validation"] = {
                "results_path": cv_path,
                "avg_profit_factor": cv_results.get("avg_profit_factor", 0),
                "avg_win_rate": cv_results.get("avg_win_rate", 0),
                "consistency_score": cv_results.get("consistency_score", 0)
            }
        
        # Determine overall success
        component_success_rate = (success_count / len(component_names)) * 100
        
        cv_success = True
        if "cross_validation" in results:
            cv_success = (
                results["cross_validation"]["avg_profit_factor"] >= 45.0 and
                results["cross_validation"]["avg_win_rate"] >= 53.0 and
                results["cross_validation"]["consistency_score"] >= 75.0
            )
            
        results["overall_success"] = (
            component_success_rate >= 75.0 and
            cv_success
        )
        
        # Save overall results
        overall_path = os.path.join(output_dir, f"{timestamp}_overall_results.json")
        with open(overall_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Log overall results
        logger.info("High-performance backtest complete")
        logger.info(f"Component success rate: {component_success_rate:.2f}%")
        logger.info(f"Overall success: {results['overall_success']}")
        logger.info(f"Results saved to {output_dir}")
        
        return results
