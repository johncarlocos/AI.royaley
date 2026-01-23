"""
ROYALEY - Betting Simulator
Monte Carlo simulation and scenario analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for betting simulation"""
    # Bankroll settings
    initial_bankroll: float = 10000.0
    kelly_fraction: float = 0.25
    max_bet_percent: float = 0.02
    
    # Simulation parameters
    num_simulations: int = 1000
    bets_per_simulation: int = 500
    
    # Performance assumptions
    win_rate: float = 0.55  # Expected win rate
    avg_odds: int = -110  # Average betting odds
    win_rate_std: float = 0.05  # Standard deviation in win rate
    
    # Risk parameters
    stop_loss_percent: Optional[float] = 0.25  # Stop if down 25%
    take_profit_percent: Optional[float] = None
    
    # Variance settings
    streak_factor: float = 0.0  # 0 = independent, >0 = more streaky
    
    # Parallel processing
    n_jobs: int = 4


@dataclass
class SimulationResult:
    """Results from betting simulation"""
    config: SimulationConfig
    
    # Summary statistics
    median_final_bankroll: float
    mean_final_bankroll: float
    std_final_bankroll: float
    
    # Percentiles
    p5_final_bankroll: float   # 5th percentile (worst case)
    p25_final_bankroll: float  # 25th percentile
    p75_final_bankroll: float  # 75th percentile
    p95_final_bankroll: float  # 95th percentile (best case)
    
    # Risk metrics
    probability_of_ruin: float  # % of simulations hitting stop loss
    probability_of_profit: float  # % of simulations profitable
    max_drawdown_median: float
    max_drawdown_95: float  # 95th percentile max drawdown
    
    # Performance metrics
    median_roi: float
    expected_value: float
    sharpe_ratio: float
    
    # Distribution data
    final_bankrolls: List[float] = field(default_factory=list)
    roi_distribution: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    
    # Time to goals
    time_to_double_median: Optional[int] = None
    time_to_double_probability: float = 0.0


@dataclass 
class ScenarioResult:
    """Results from scenario analysis"""
    scenario_name: str
    win_rate: float
    avg_odds: int
    kelly_fraction: float
    
    # Metrics
    expected_roi: float
    probability_of_profit: float
    probability_of_ruin: float
    median_final_bankroll: float
    max_drawdown_95: float
    
    # Assessment
    edge: float
    recommended: bool
    notes: str


class BettingSimulator:
    """
    Monte Carlo simulation for betting strategies
    """
    
    def __init__(self):
        self.rng = np.random.default_rng()
    
    def run_simulation(self, config: SimulationConfig) -> SimulationResult:
        """
        Run Monte Carlo simulation
        
        Args:
            config: Simulation configuration
            
        Returns:
            SimulationResult with comprehensive metrics
        """
        logger.info(f"Starting Monte Carlo simulation with {config.num_simulations} iterations")
        
        final_bankrolls = []
        max_drawdowns = []
        time_to_double = []
        
        # Run simulations
        if config.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
                futures = [
                    executor.submit(self._run_single_simulation, config)
                    for _ in range(config.num_simulations)
                ]
                
                for future in futures:
                    result = future.result()
                    final_bankrolls.append(result['final_bankroll'])
                    max_drawdowns.append(result['max_drawdown'])
                    if result['time_to_double']:
                        time_to_double.append(result['time_to_double'])
        else:
            for _ in range(config.num_simulations):
                result = self._run_single_simulation(config)
                final_bankrolls.append(result['final_bankroll'])
                max_drawdowns.append(result['max_drawdown'])
                if result['time_to_double']:
                    time_to_double.append(result['time_to_double'])
        
        # Calculate statistics
        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)
        
        roi_distribution = [(fb - config.initial_bankroll) / config.initial_bankroll * 100 
                           for fb in final_bankrolls]
        
        # Calculate metrics
        median_final = float(np.median(final_bankrolls))
        mean_final = float(np.mean(final_bankrolls))
        std_final = float(np.std(final_bankrolls))
        
        p5 = float(np.percentile(final_bankrolls, 5))
        p25 = float(np.percentile(final_bankrolls, 25))
        p75 = float(np.percentile(final_bankrolls, 75))
        p95 = float(np.percentile(final_bankrolls, 95))
        
        # Risk metrics
        stop_loss_threshold = config.initial_bankroll * (1 - config.stop_loss_percent) if config.stop_loss_percent else 0
        prob_ruin = np.mean(final_bankrolls <= stop_loss_threshold) * 100
        prob_profit = np.mean(final_bankrolls > config.initial_bankroll) * 100
        
        median_drawdown = float(np.median(max_drawdowns)) * 100
        p95_drawdown = float(np.percentile(max_drawdowns, 95)) * 100
        
        # Performance metrics
        median_roi = (median_final - config.initial_bankroll) / config.initial_bankroll * 100
        expected_value = self._calculate_expected_value(config)
        
        # Sharpe ratio approximation
        returns = np.diff(final_bankrolls) / final_bankrolls[:-1] if len(final_bankrolls) > 1 else [0]
        sharpe = (np.mean(roi_distribution) / np.std(roi_distribution)) if np.std(roi_distribution) > 0 else 0
        
        # Time to double
        time_to_double_median = int(np.median(time_to_double)) if time_to_double else None
        time_to_double_prob = len(time_to_double) / config.num_simulations * 100
        
        return SimulationResult(
            config=config,
            median_final_bankroll=median_final,
            mean_final_bankroll=mean_final,
            std_final_bankroll=std_final,
            p5_final_bankroll=p5,
            p25_final_bankroll=p25,
            p75_final_bankroll=p75,
            p95_final_bankroll=p95,
            probability_of_ruin=prob_ruin,
            probability_of_profit=prob_profit,
            max_drawdown_median=median_drawdown,
            max_drawdown_95=p95_drawdown,
            median_roi=median_roi,
            expected_value=expected_value,
            sharpe_ratio=sharpe,
            final_bankrolls=list(final_bankrolls),
            roi_distribution=roi_distribution,
            max_drawdowns=list(max_drawdowns),
            time_to_double_median=time_to_double_median,
            time_to_double_probability=time_to_double_prob
        )
    
    def _run_single_simulation(self, config: SimulationConfig) -> Dict:
        """Run a single simulation iteration"""
        bankroll = config.initial_bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0
        time_to_double = None
        
        # Simulate variable win rate
        sim_win_rate = np.clip(
            np.random.normal(config.win_rate, config.win_rate_std),
            0.4, 0.7  # Reasonable bounds
        )
        
        # Previous result for streak simulation
        prev_win = None
        
        for bet_num in range(config.bets_per_simulation):
            # Check stop conditions
            if config.stop_loss_percent:
                stop_threshold = config.initial_bankroll * (1 - config.stop_loss_percent)
                if bankroll <= stop_threshold:
                    break
            
            if config.take_profit_percent:
                profit_threshold = config.initial_bankroll * (1 + config.take_profit_percent)
                if bankroll >= profit_threshold:
                    break
            
            # Calculate bet size
            bet_size = self._calculate_bet_size(
                bankroll,
                sim_win_rate,
                config.avg_odds,
                config.kelly_fraction,
                config.max_bet_percent
            )
            
            if bet_size < 10:  # Minimum bet
                bet_size = min(10, bankroll * 0.01)
            
            if bet_size > bankroll:
                break
            
            # Determine outcome
            win_prob = sim_win_rate
            
            # Apply streak factor
            if config.streak_factor > 0 and prev_win is not None:
                if prev_win:
                    win_prob = sim_win_rate + config.streak_factor * 0.1
                else:
                    win_prob = sim_win_rate - config.streak_factor * 0.1
                win_prob = np.clip(win_prob, 0.3, 0.8)
            
            won = random.random() < win_prob
            prev_win = won
            
            # Update bankroll
            if won:
                profit = self._calculate_profit(bet_size, config.avg_odds)
                bankroll += profit
            else:
                bankroll -= bet_size
            
            # Track peak and drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            
            current_drawdown = (peak_bankroll - bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Check time to double
            if time_to_double is None and bankroll >= config.initial_bankroll * 2:
                time_to_double = bet_num + 1
        
        return {
            'final_bankroll': bankroll,
            'max_drawdown': max_drawdown,
            'time_to_double': time_to_double
        }
    
    def _calculate_bet_size(
        self,
        bankroll: float,
        win_rate: float,
        odds: int,
        kelly_fraction: float,
        max_bet_percent: float
    ) -> float:
        """Calculate bet size using Kelly criterion"""
        decimal_odds = self._american_to_decimal(odds)
        b = decimal_odds - 1
        p = win_rate
        q = 1 - p
        
        if b <= 0:
            return 0
        
        full_kelly = (b * p - q) / b
        
        if full_kelly <= 0:
            return 0
        
        kelly = full_kelly * kelly_fraction
        kelly = min(kelly, max_bet_percent)
        
        return bankroll * kelly
    
    def _calculate_profit(self, stake: float, odds: int) -> float:
        """Calculate profit from winning bet"""
        if odds > 0:
            return stake * (odds / 100)
        else:
            return stake * (100 / abs(odds))
    
    def _american_to_decimal(self, american: int) -> float:
        """Convert American odds to decimal"""
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))
    
    def _calculate_expected_value(self, config: SimulationConfig) -> float:
        """Calculate expected value per bet"""
        decimal_odds = self._american_to_decimal(config.avg_odds)
        ev = (config.win_rate * (decimal_odds - 1)) - (1 - config.win_rate)
        return ev * 100  # As percentage
    
    def run_scenario_analysis(
        self,
        base_config: SimulationConfig,
        scenarios: List[Dict[str, Any]]
    ) -> List[ScenarioResult]:
        """
        Run multiple scenarios for comparison
        
        Args:
            base_config: Base simulation configuration
            scenarios: List of scenario modifications
            
        Returns:
            List of ScenarioResult for each scenario
        """
        results = []
        
        for scenario in scenarios:
            # Create modified config
            config = SimulationConfig(
                initial_bankroll=scenario.get('initial_bankroll', base_config.initial_bankroll),
                kelly_fraction=scenario.get('kelly_fraction', base_config.kelly_fraction),
                max_bet_percent=scenario.get('max_bet_percent', base_config.max_bet_percent),
                num_simulations=scenario.get('num_simulations', min(500, base_config.num_simulations)),
                bets_per_simulation=scenario.get('bets_per_simulation', base_config.bets_per_simulation),
                win_rate=scenario.get('win_rate', base_config.win_rate),
                avg_odds=scenario.get('avg_odds', base_config.avg_odds),
                win_rate_std=scenario.get('win_rate_std', base_config.win_rate_std),
                stop_loss_percent=scenario.get('stop_loss_percent', base_config.stop_loss_percent),
                n_jobs=1  # Single thread for scenarios
            )
            
            # Run simulation
            sim_result = self.run_simulation(config)
            
            # Calculate edge
            implied_prob = abs(config.avg_odds) / (abs(config.avg_odds) + 100) if config.avg_odds < 0 else 100 / (config.avg_odds + 100)
            edge = config.win_rate - implied_prob
            
            # Assess recommendation
            recommended = (
                sim_result.probability_of_profit > 60 and
                sim_result.probability_of_ruin < 10 and
                sim_result.median_roi > 0
            )
            
            notes = []
            if sim_result.probability_of_ruin > 20:
                notes.append("High ruin risk")
            if sim_result.max_drawdown_95 > 40:
                notes.append("Large potential drawdowns")
            if edge < 0.02:
                notes.append("Thin edge margin")
            if sim_result.median_roi > 20:
                notes.append("Strong expected returns")
            
            results.append(ScenarioResult(
                scenario_name=scenario.get('name', 'Unnamed'),
                win_rate=config.win_rate,
                avg_odds=config.avg_odds,
                kelly_fraction=config.kelly_fraction,
                expected_roi=sim_result.median_roi,
                probability_of_profit=sim_result.probability_of_profit,
                probability_of_ruin=sim_result.probability_of_ruin,
                median_final_bankroll=sim_result.median_final_bankroll,
                max_drawdown_95=sim_result.max_drawdown_95,
                edge=edge * 100,
                recommended=recommended,
                notes="; ".join(notes) if notes else "Acceptable parameters"
            ))
        
        return results
    
    def sensitivity_analysis(
        self,
        base_config: SimulationConfig,
        parameter: str,
        values: List[Any]
    ) -> Dict[str, List]:
        """
        Run sensitivity analysis on a single parameter
        
        Args:
            base_config: Base configuration
            parameter: Parameter to vary
            values: Values to test
            
        Returns:
            Dict with parameter values and corresponding metrics
        """
        results = {
            'parameter': parameter,
            'values': values,
            'median_roi': [],
            'prob_profit': [],
            'prob_ruin': [],
            'max_drawdown': []
        }
        
        for value in values:
            # Create modified config
            config_dict = {
                'initial_bankroll': base_config.initial_bankroll,
                'kelly_fraction': base_config.kelly_fraction,
                'max_bet_percent': base_config.max_bet_percent,
                'num_simulations': 200,  # Reduced for sensitivity
                'bets_per_simulation': base_config.bets_per_simulation,
                'win_rate': base_config.win_rate,
                'avg_odds': base_config.avg_odds,
                'win_rate_std': base_config.win_rate_std,
                'stop_loss_percent': base_config.stop_loss_percent,
                'n_jobs': 1
            }
            config_dict[parameter] = value
            config = SimulationConfig(**config_dict)
            
            sim_result = self.run_simulation(config)
            
            results['median_roi'].append(sim_result.median_roi)
            results['prob_profit'].append(sim_result.probability_of_profit)
            results['prob_ruin'].append(sim_result.probability_of_ruin)
            results['max_drawdown'].append(sim_result.max_drawdown_95)
        
        return results


class KellyOptimizer:
    """Optimize Kelly fraction for given parameters"""
    
    def __init__(self, simulator: BettingSimulator):
        self.simulator = simulator
    
    def find_optimal_kelly(
        self,
        win_rate: float,
        avg_odds: int,
        initial_bankroll: float = 10000,
        target_metric: str = 'sharpe',
        kelly_range: Tuple[float, float] = (0.1, 0.5),
        steps: int = 9
    ) -> Dict:
        """
        Find optimal Kelly fraction
        
        Args:
            win_rate: Expected win rate
            avg_odds: Average betting odds
            initial_bankroll: Starting bankroll
            target_metric: Metric to optimize ('sharpe', 'roi', 'risk_adjusted')
            kelly_range: Range of Kelly fractions to test
            steps: Number of values to test
            
        Returns:
            Dict with optimal Kelly and analysis
        """
        kelly_values = np.linspace(kelly_range[0], kelly_range[1], steps)
        results = []
        
        for kelly in kelly_values:
            config = SimulationConfig(
                initial_bankroll=initial_bankroll,
                kelly_fraction=kelly,
                win_rate=win_rate,
                avg_odds=avg_odds,
                num_simulations=300,
                bets_per_simulation=500,
                n_jobs=1
            )
            
            sim_result = self.simulator.run_simulation(config)
            
            # Calculate risk-adjusted return
            if sim_result.max_drawdown_95 > 0:
                risk_adjusted = sim_result.median_roi / sim_result.max_drawdown_95
            else:
                risk_adjusted = sim_result.median_roi
            
            results.append({
                'kelly': kelly,
                'roi': sim_result.median_roi,
                'sharpe': sim_result.sharpe_ratio,
                'risk_adjusted': risk_adjusted,
                'prob_ruin': sim_result.probability_of_ruin,
                'max_drawdown': sim_result.max_drawdown_95
            })
        
        # Find optimal
        if target_metric == 'sharpe':
            optimal = max(results, key=lambda x: x['sharpe'])
        elif target_metric == 'roi':
            optimal = max(results, key=lambda x: x['roi'])
        else:  # risk_adjusted
            optimal = max(results, key=lambda x: x['risk_adjusted'])
        
        return {
            'optimal_kelly': optimal['kelly'],
            'optimal_roi': optimal['roi'],
            'optimal_sharpe': optimal['sharpe'],
            'risk_adjusted_return': optimal['risk_adjusted'],
            'probability_of_ruin': optimal['prob_ruin'],
            'max_drawdown_95': optimal['max_drawdown'],
            'all_results': results
        }
