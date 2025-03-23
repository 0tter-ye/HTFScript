import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add parent directory to path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ultra_optimized_strategy import UltraOptimizedStrategy

# Try different import approaches for ultra_optimized_backtest
try:
    # First try direct import if running as a module
    from ultra_optimized_backtest import UltraOptimizedBacktest
except ImportError:
    # If that fails, try importing with full path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ultra_optimized_backtest import UltraOptimizedBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'advanced_components_test.log'))
    ]
)

logger = logging.getLogger('advanced_components_test')

def run_advanced_components_test(symbols=None, timeframes=None, days=5, test_periods=None):
    """
    Run a test to verify that the advanced quantitative components are working correctly
    
    Args:
        symbols (list): List of symbols to test
        timeframes (list): List of timeframes to test
        days (int): Number of days to test
        test_periods (dict): Dictionary of test periods in days, e.g. {'1month': 30, '6month': 180, '1year': 365, '5year': 1825}
        
    Returns:
        dict: Test results
    """
    # Set default symbols and timeframes
    if symbols is None:
        symbols = ["BTCUSDT"]
    
    if timeframes is None:
        timeframes = ["5m", "15m", "1h"]
        
    if test_periods is None:
        test_periods = {
            '1month': 30,
            '6month': 180, 
            '1year': 365,
            '5year': 1825
        }
    
    logger.info(f"Starting advanced components test with {len(symbols)} symbols and {len(timeframes)} timeframes")
    logger.info(f"Test period: {days} days")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Initialize results dictionary
    results = {
        'by_component': {},
        'by_timeframe': {},
        'by_symbol': {},
        'overall': {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'profit_factor': 0,
            'win_rate': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'return_multiplier': 0,
            'avg_trade_return': 0,
            'return_std_dev': 0,
            'last_trade_z_score': 0,
            'markov_entropy': 0,
            'markov_win_probability': 0,
            'decision_tree_accuracy': 0,
            'decision_tree_precision': 0,
            'decision_tree_recall': 0,
            'decision_tree_f1': 0,
            'decision_tree_win_probability': 0,
            'component_usage': {}
        }
    }
    
    # Initialize component tracking
    component_names = [
        'price_action', 'volume_profile', 'support_resistance', 'volatility_analysis',
        'momentum_divergence', 'order_flow', 'market_regime', 'correlation_analysis',
        'ml_assessment', 'random_forest', 'rnn_strategy', 'microstructure_noise',
        'bid_ask_imbalance', 'fractal_dimension', 'hurst_exponent'
    ]
    
    for component in component_names:
        results['by_component'][component] = {
            'usage_count': 0,
            'signal_count': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'profit': 0
        }
    
    # Run tests for each symbol and timeframe
    for symbol in symbols:
        results['by_symbol'][symbol] = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'profit_factor': 0,
            'win_rate': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'return_multiplier': 0,
            'avg_trade_return': 0,
            'return_std_dev': 0,
            'last_trade_z_score': 0,
            'markov_entropy': 0,
            'markov_win_probability': 0,
            'decision_tree_accuracy': 0,
            'decision_tree_precision': 0,
            'decision_tree_recall': 0,
            'decision_tree_f1': 0,
            'decision_tree_win_probability': 0
        }
        
        for timeframe in timeframes:
            logger.info(f"Testing {symbol} on {timeframe} timeframe")
            
            # Initialize strategy
            strategy = UltraOptimizedStrategy(
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=10000,
                risk_per_trade=0.02,
                max_position_size=100.0,
                backtest_days=days
            )
            
            # Initialize backtest runner
            backtest_runner = UltraOptimizedBacktest(
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=10000,
                max_position_size=100.0
            )
            
            # Set date range for backtest
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            backtest_runner.strategy.start_date = start_date.strftime('%Y-%m-%d')
            backtest_runner.strategy.end_date = end_date.strftime('%Y-%m-%d')
            
            # Enable advanced components
            backtest_runner.strategy.params.update({
                'use_markov_chains': True,
                'use_ml_models': True,
                'use_fractal_analysis': True,
                'use_microstructure': True,
                'use_volume_analysis': True,
                'use_rnn_strategy_selection': True,
                'use_bid_ask_imbalance': True,
                'use_support_resistance': True,
                'use_volatility_analysis': True,
                'use_momentum_divergence': True,
                'use_order_flow': True,
                'use_market_regime': True,
                'use_correlation_analysis': True,
                'market_regime_lookback': 30,
                'market_regime_weight': 2.0,
                'markov_confidence_weight': 2.0,
                'ml_confidence_weight': 1.5,
                'fractal_confidence_weight': 1.8,
                'price_action_weight': 1.0,
                'volume_confidence_weight': 1.5
            })
            
            # Run backtest
            backtest_results = backtest_runner.run()
            
            # Handle case where backtest_results might be None or have an error
            if backtest_results is None or 'error' in backtest_results:
                logger.error(f"Error running backtest for {symbol} on {timeframe}: {backtest_results.get('error', 'Unknown error')}")
                continue
                
            # Extract metrics from backtest results
            metrics = backtest_results.get('metrics', {})
            
            # Extract component usage from strategy
            for component in component_names:
                component_attr = f"{component}_usage"
                if hasattr(backtest_runner.strategy, component_attr):
                    usage_count = getattr(backtest_runner.strategy, component_attr)
                    results['by_component'][component]['usage_count'] += usage_count
                    
                    if component_attr not in results['overall']['component_usage']:
                        results['overall']['component_usage'][component_attr] = 0
                    results['overall']['component_usage'][component_attr] += usage_count
            
            # Calculate and update Sharpe ratio for this symbol/timeframe
            if 'sharpe_ratio' in metrics:
                # Add Sharpe ratio to symbol results
                results['by_symbol'][symbol]['sharpe_ratio'] = metrics['sharpe_ratio']
                
                # Add to timeframe results
                if 'sharpe_ratio' not in results['by_timeframe'][timeframe]:
                    results['by_timeframe'][timeframe]['sharpe_ratio'] = 0
                
                # Use weighted average based on number of trades
                current_weight = results['by_timeframe'][timeframe]['total_trades']
                new_weight = len(backtest_results.get('trades', []))
                total_weight = current_weight + new_weight
                
                if total_weight > 0:
                    results['by_timeframe'][timeframe]['sharpe_ratio'] = (
                        (results['by_timeframe'][timeframe]['sharpe_ratio'] * current_weight) + 
                        (metrics['sharpe_ratio'] * new_weight)
                    ) / total_weight
                
                # Add to overall results
                if 'sharpe_ratio' not in results['overall']:
                    results['overall']['sharpe_ratio'] = 0
                
                current_weight = results['overall']['total_trades'] - len(backtest_results.get('trades', []))
                new_weight = len(backtest_results.get('trades', []))
                total_weight = current_weight + new_weight
                
                if total_weight > 0:
                    results['overall']['sharpe_ratio'] = (
                        (results['overall']['sharpe_ratio'] * current_weight) + 
                        (metrics['sharpe_ratio'] * new_weight)
                    ) / total_weight
            
            # Extract additional performance metrics
            if 'return_pct' in metrics:
                return_multiplier = 1 + metrics.get('return_pct', 0)
                results['by_symbol'][symbol]['return_multiplier'] = return_multiplier
            
            if 'average_win_pct' in metrics and 'average_loss_pct' in metrics:
                # Calculate average trade return across all trades
                avg_trade_return = 0
                if results['by_symbol'][symbol]['total_trades'] > 0:
                    total_return = (metrics.get('average_win_pct', 0) * results['by_symbol'][symbol]['winning_trades'] + 
                                   metrics.get('average_loss_pct', 0) * results['by_symbol'][symbol]['losing_trades'])
                    avg_trade_return = total_return / results['by_symbol'][symbol]['total_trades']
                results['by_symbol'][symbol]['avg_trade_return'] = avg_trade_return
            
            # Extract standard deviation of returns if available
            if 'return_std_dev' in metrics:
                results['by_symbol'][symbol]['return_std_dev'] = metrics.get('return_std_dev', 0)
            
            # Extract last trade Z-score if available
            if 'last_trade_z_score' in metrics:
                results['by_symbol'][symbol]['last_trade_z_score'] = metrics.get('last_trade_z_score', 0)
            
            # Extract Markov chain metrics if available
            if hasattr(backtest_runner.strategy, 'markov_chain_entropy'):
                results['by_symbol'][symbol]['markov_entropy'] = backtest_runner.strategy.markov_chain_entropy
            
            if hasattr(backtest_runner.strategy, 'markov_win_probability'):
                results['by_symbol'][symbol]['markov_win_probability'] = backtest_runner.strategy.markov_win_probability
            
            # Extract decision tree metrics if available
            if hasattr(backtest_runner.strategy, 'decision_tree_metrics'):
                dt_metrics = backtest_runner.strategy.decision_tree_metrics
                results['by_symbol'][symbol]['decision_tree_accuracy'] = dt_metrics.get('accuracy', 0)
                results['by_symbol'][symbol]['decision_tree_precision'] = dt_metrics.get('precision', 0)
                results['by_symbol'][symbol]['decision_tree_recall'] = dt_metrics.get('recall', 0)
                results['by_symbol'][symbol]['decision_tree_f1'] = dt_metrics.get('f1', 0)
                results['by_symbol'][symbol]['decision_tree_win_probability'] = dt_metrics.get('win_probability', 0)
            
            # Update results
            if timeframe not in results['by_timeframe']:
                results['by_timeframe'][timeframe] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
                
            # Update symbol results
            symbol_trades = len(backtest_results.get('trades', []))
            symbol_winning_trades = sum(1 for trade in backtest_results.get('trades', []) if trade.get('profit_amount', 0) > 0)
            symbol_losing_trades = sum(1 for trade in backtest_results.get('trades', []) if trade.get('profit_amount', 0) <= 0)
            symbol_profit = sum(trade.get('profit_amount', 0) for trade in backtest_results.get('trades', []))
            
            # Calculate max drawdown from equity curve if available
            max_drawdown = metrics.get('max_drawdown', 0)
            if 'equity_curve' in backtest_results and len(backtest_results['equity_curve']) > 0:
                equity_curve = backtest_results['equity_curve']
                max_equity = max(equity_curve)
                current_drawdown = 0
                for equity in equity_curve:
                    if equity < max_equity:
                        current_drawdown = (max_equity - equity) / max_equity * 100
                        if current_drawdown > max_drawdown:
                            max_drawdown = current_drawdown
                    else:
                        max_equity = equity
            
            # Update symbol metrics
            results['by_symbol'][symbol]['total_trades'] += symbol_trades
            results['by_symbol'][symbol]['winning_trades'] += symbol_winning_trades
            results['by_symbol'][symbol]['losing_trades'] += symbol_losing_trades
            results['by_symbol'][symbol]['total_profit'] += symbol_profit
            
            if symbol_trades > 0:
                results['by_symbol'][symbol]['win_rate'] = symbol_winning_trades / symbol_trades * 100
            
            if symbol_losing_trades > 0 and symbol_winning_trades > 0:
                avg_win = sum(trade.get('profit_amount', 0) for trade in backtest_results.get('trades', []) if trade.get('profit_amount', 0) > 0) / symbol_winning_trades
                avg_loss = abs(sum(trade.get('profit_amount', 0) for trade in backtest_results.get('trades', []) if trade.get('profit_amount', 0) <= 0)) / symbol_losing_trades
                if avg_loss > 0:
                    results['by_symbol'][symbol]['profit_factor'] = avg_win / avg_loss
            
            # Update max drawdown
            if max_drawdown > results['by_symbol'][symbol]['max_drawdown']:
                results['by_symbol'][symbol]['max_drawdown'] = max_drawdown
            
            # Update timeframe metrics
            results['by_timeframe'][timeframe]['total_trades'] += symbol_trades
            results['by_timeframe'][timeframe]['winning_trades'] += symbol_winning_trades
            results['by_timeframe'][timeframe]['losing_trades'] += symbol_losing_trades
            results['by_timeframe'][timeframe]['total_profit'] += symbol_profit
            
            if results['by_timeframe'][timeframe]['total_trades'] > 0:
                results['by_timeframe'][timeframe]['win_rate'] = results['by_timeframe'][timeframe]['winning_trades'] / results['by_timeframe'][timeframe]['total_trades'] * 100
            
            # Update max drawdown for timeframe
            if max_drawdown > results['by_timeframe'][timeframe]['max_drawdown']:
                results['by_timeframe'][timeframe]['max_drawdown'] = max_drawdown
            
            # Update timeframe profit factor
            timeframe_winning_trades = sum(trade.get('profit_amount', 0) for trade in backtest_results.get('trades', []) if trade.get('profit_amount', 0) > 0)
            timeframe_losing_trades = abs(sum(trade.get('profit_amount', 0) for trade in backtest_results.get('trades', []) if trade.get('profit_amount', 0) <= 0))
            
            if timeframe_losing_trades > 0:
                results['by_timeframe'][timeframe]['profit_factor'] = timeframe_winning_trades / timeframe_losing_trades
            elif timeframe_winning_trades > 0:
                results['by_timeframe'][timeframe]['profit_factor'] = float('inf')
            
            # Update overall metrics
            results['overall']['total_trades'] += symbol_trades
            results['overall']['winning_trades'] += symbol_winning_trades
            results['overall']['losing_trades'] += symbol_losing_trades
            results['overall']['total_profit'] += symbol_profit
            
            if results['overall']['total_trades'] > 0:
                results['overall']['win_rate'] = results['overall']['winning_trades'] / results['overall']['total_trades'] * 100
            
            # Update max drawdown for overall
            if max_drawdown > results['overall']['max_drawdown']:
                results['overall']['max_drawdown'] = max_drawdown
            
            # Log results for this symbol and timeframe
            logger.info(f"Results for {symbol} on {timeframe}:")
            logger.info(f"  Total trades: {symbol_trades}")
            logger.info(f"  Win rate: {symbol_winning_trades / symbol_trades * 100:.2f}% ({symbol_winning_trades}/{symbol_trades})" if symbol_trades > 0 else "  Win rate: N/A (0 trades)")
            logger.info(f"  Total profit: {symbol_profit:.2f}")
            logger.info(f"  Max drawdown: {max_drawdown:.2f}%")
            logger.info(f"  Sharpe ratio: {results['by_symbol'][symbol].get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Return multiplier: {results['by_symbol'][symbol].get('return_multiplier', 0):.2f}x")
            logger.info(f"  Average trade return: {results['by_symbol'][symbol].get('avg_trade_return', 0)*100:.2f}%")
            logger.info(f"  Return std dev: {results['by_symbol'][symbol].get('return_std_dev', 0)*100:.2f}%")
            
            # Log Markov chain metrics if available
            if 'markov_entropy' in results['by_symbol'][symbol]:
                logger.info(f"  Markov chain entropy: {results['by_symbol'][symbol]['markov_entropy']:.2f} bits")
                logger.info(f"  Markov win probability: {results['by_symbol'][symbol].get('markov_win_probability', 0)*100:.2f}%")
            
            # Log decision tree metrics if available
            if 'decision_tree_accuracy' in results['by_symbol'][symbol]:
                logger.info(f"  Decision tree accuracy: {results['by_symbol'][symbol]['decision_tree_accuracy']:.2f}")
                logger.info(f"  Decision tree precision: {results['by_symbol'][symbol]['decision_tree_precision']:.2f}")
                logger.info(f"  Decision tree recall: {results['by_symbol'][symbol]['decision_tree_recall']:.2f}")
                logger.info(f"  Decision tree F1: {results['by_symbol'][symbol]['decision_tree_f1']:.2f}")
                logger.info(f"  Decision tree win probability: {results['by_symbol'][symbol]['decision_tree_win_probability']*100:.2f}%")
    
    # Calculate overall metrics
    if results['overall']['total_trades'] > 0:
        results['overall']['win_rate'] = results['overall']['winning_trades'] / results['overall']['total_trades'] * 100
        
        total_gains = sum(results['by_symbol'][symbol]['total_profit'] for symbol in symbols if results['by_symbol'][symbol]['total_profit'] > 0)
        total_losses = abs(sum(results['by_symbol'][symbol]['total_profit'] for symbol in symbols if results['by_symbol'][symbol]['total_profit'] < 0))
        
        if total_losses > 0:
            results['overall']['profit_factor'] = total_gains / total_losses
        else:
            results['overall']['profit_factor'] = float('inf') if total_gains > 0 else 0
            
        # Calculate overall return multiplier (weighted average based on trades)
        total_trades = results['overall']['total_trades']
        if total_trades > 0:
            results['overall']['return_multiplier'] = sum(
                results['by_symbol'][symbol].get('return_multiplier', 1.0) * results['by_symbol'][symbol]['total_trades'] 
                for symbol in symbols
            ) / total_trades
            
            # Calculate overall average trade return (weighted average)
            results['overall']['avg_trade_return'] = sum(
                results['by_symbol'][symbol].get('avg_trade_return', 0) * results['by_symbol'][symbol]['total_trades']
                for symbol in symbols
            ) / total_trades
            
            # Calculate overall return standard deviation (weighted average)
            results['overall']['return_std_dev'] = sum(
                results['by_symbol'][symbol].get('return_std_dev', 0) * results['by_symbol'][symbol]['total_trades']
                for symbol in symbols
            ) / total_trades
            
        # Calculate overall Markov metrics (weighted average)
        markov_symbols = [s for s in symbols if 'markov_entropy' in results['by_symbol'][s]]
        if markov_symbols:
            markov_trades = sum(results['by_symbol'][s]['total_trades'] for s in markov_symbols)
            if markov_trades > 0:
                results['overall']['markov_entropy'] = sum(
                    results['by_symbol'][s]['markov_entropy'] * results['by_symbol'][s]['total_trades']
                    for s in markov_symbols
                ) / markov_trades
                
                results['overall']['markov_win_probability'] = sum(
                    results['by_symbol'][s].get('markov_win_probability', 0) * results['by_symbol'][s]['total_trades']
                    for s in markov_symbols
                ) / markov_trades
                
        # Calculate overall decision tree metrics (weighted average)
        dt_symbols = [s for s in symbols if 'decision_tree_accuracy' in results['by_symbol'][s]]
        if dt_symbols:
            dt_trades = sum(results['by_symbol'][s]['total_trades'] for s in dt_symbols)
            if dt_trades > 0:
                for metric in ['decision_tree_accuracy', 'decision_tree_precision', 'decision_tree_recall', 
                              'decision_tree_f1', 'decision_tree_win_probability']:
                    results['overall'][metric] = sum(
                        results['by_symbol'][s].get(metric, 0) * results['by_symbol'][s]['total_trades']
                        for s in dt_symbols
                    ) / dt_trades
    
    # Log overall results
    logger.info("\nOverall Results:")
    logger.info(f"Total trades: {results['overall']['total_trades']}")
    logger.info(f"Win rate: {results['overall']['win_rate']:.2f}% ({results['overall']['winning_trades']}/{results['overall']['total_trades']})" if results['overall']['total_trades'] > 0 else "Win rate: N/A (0 trades)")
    logger.info(f"Profit factor: {results['overall']['profit_factor']:.2f}" if results['overall']['profit_factor'] != float('inf') else "Profit factor: Infinite")
    logger.info(f"Total profit: {results['overall']['total_profit']:.2f}")
    logger.info(f"Max drawdown: {results['overall']['max_drawdown']:.2f}%")
    logger.info(f"Sharpe ratio: {results['overall']['sharpe_ratio']:.2f}")
    logger.info(f"Return multiplier: {results['overall']['return_multiplier']:.2f}x")
    logger.info(f"Average trade return: {results['overall']['avg_trade_return']*100:.2f}%")
    logger.info(f"Return std dev: {results['overall']['return_std_dev']*100:.2f}%")
    
    # Log Markov chain metrics if available
    if results['overall'].get('markov_entropy', 0) != 0:
        logger.info(f"Markov chain entropy: {results['overall']['markov_entropy']:.2f} bits")
        logger.info(f"Markov win probability: {results['overall']['markov_win_probability']*100:.2f}%")
    
    # Log decision tree metrics if available
    if results['overall'].get('decision_tree_accuracy', 0) != 0:
        logger.info(f"Decision tree accuracy: {results['overall']['decision_tree_accuracy']:.2f}")
        logger.info(f"Decision tree precision: {results['overall']['decision_tree_precision']:.2f}")
        logger.info(f"Decision tree recall: {results['overall']['decision_tree_recall']:.2f}")
        logger.info(f"Decision tree F1: {results['overall']['decision_tree_f1']:.2f}")
        logger.info(f"Decision tree win probability: {results['overall']['decision_tree_win_probability']*100:.2f}%")
    
    # Log component usage
    logger.info("\nComponent Usage:")
    for component, usage in results['by_component'].items():
        logger.info(f"{component}: {usage['usage_count']} times")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Symbol': list(results['by_symbol'].keys()) + ['Overall'],
        'Total Trades': [results['by_symbol'][symbol]['total_trades'] for symbol in results['by_symbol']] + [results['overall']['total_trades']],
        'Win Rate': [results['by_symbol'][symbol]['win_rate'] if results['by_symbol'][symbol]['total_trades'] > 0 else 0 for symbol in results['by_symbol']] + [results['overall']['win_rate'] if results['overall']['total_trades'] > 0 else 0],
        'Profit Factor': [results['by_symbol'][symbol]['profit_factor'] if results['by_symbol'][symbol]['profit_factor'] != float('inf') else 999 for symbol in results['by_symbol']] + [results['overall']['profit_factor'] if results['overall']['profit_factor'] != float('inf') else 999],
        'Total Profit': [results['by_symbol'][symbol]['total_profit'] for symbol in results['by_symbol']] + [results['overall']['total_profit']],
        'Max Drawdown': [results['by_symbol'][symbol]['max_drawdown'] for symbol in results['by_symbol']] + [results['overall']['max_drawdown']],
        'Sharpe Ratio': [results['by_symbol'][symbol]['sharpe_ratio'] for symbol in results['by_symbol']] + [results['overall']['sharpe_ratio']],
        'Return Multiplier': [results['by_symbol'][symbol]['return_multiplier'] for symbol in results['by_symbol']] + [results['overall']['return_multiplier']],
        'Average Trade Return': [results['by_symbol'][symbol]['avg_trade_return'] for symbol in results['by_symbol']] + [results['overall']['avg_trade_return']],
        'Return Standard Deviation': [results['by_symbol'][symbol]['return_std_dev'] for symbol in results['by_symbol']] + [results['overall']['return_std_dev']],
        'Last Trade Z-Score': [results['by_symbol'][symbol]['last_trade_z_score'] for symbol in results['by_symbol']] + [results['overall']['last_trade_z_score']],
        'Markov Entropy': [results['by_symbol'][symbol]['markov_entropy'] for symbol in results['by_symbol']] + [results['overall']['markov_entropy']],
        'Markov Win Probability': [results['by_symbol'][symbol]['markov_win_probability'] for symbol in results['by_symbol']] + [results['overall']['markov_win_probability']],
        'Decision Tree Accuracy': [results['by_symbol'][symbol]['decision_tree_accuracy'] for symbol in results['by_symbol']] + [results['overall']['decision_tree_accuracy']],
        'Decision Tree Precision': [results['by_symbol'][symbol]['decision_tree_precision'] for symbol in results['by_symbol']] + [results['overall']['decision_tree_precision']],
        'Decision Tree Recall': [results['by_symbol'][symbol]['decision_tree_recall'] for symbol in results['by_symbol']] + [results['overall']['decision_tree_recall']],
        'Decision Tree F1': [results['by_symbol'][symbol]['decision_tree_f1'] for symbol in results['by_symbol']] + [results['overall']['decision_tree_f1']],
        'Decision Tree Win Probability': [results['by_symbol'][symbol]['decision_tree_win_probability'] for symbol in results['by_symbol']] + [results['overall']['decision_tree_win_probability']]
    })
    
    results_df.to_csv(os.path.join(results_dir, 'advanced_components_test_results.csv'), index=False)
    
    # Create component usage chart
    if results['overall']['component_usage']:
        plt.figure(figsize=(12, 8))
        components = list(results['by_component'].keys())
        usage_counts = [results['by_component'][component]['usage_count'] for component in components]
        
        plt.bar(components, usage_counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Component Usage in Trading Decisions')
        plt.xlabel('Component')
        plt.ylabel('Usage Count')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'component_usage.png'))
    
    return results

if __name__ == "__main__":
    # Run test with default parameters
    results = run_advanced_components_test()
    
    print("\nTest completed. Results saved to the 'results' directory.")
