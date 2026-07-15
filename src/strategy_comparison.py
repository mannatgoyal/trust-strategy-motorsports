import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.monte_carlo import F1MonteCarloSimulator

class F1StrategyComparisonEngine:
    """
    F1 Strategy Comparison Engine.
    Compares 1-Stop vs. 2-Stop strategy profiles by executing joint Monte Carlo 
    simulation runs, evaluating expected race times, volatility risk, 
    and outputting tactical recommendations.
    """
    def __init__(self, data: pd.DataFrame, sc_probs: np.ndarray):
        self.data = data.copy()
        self.sc_probs = sc_probs
        self.simulator = F1MonteCarloSimulator(data)
        
    def generate_strategy_profiles(self, pit_laps: list, pace_level: float) -> np.ndarray:
        """
        Creates a pacing strategy vector. Pit laps are denoted by a pace level of 0.10.
        """
        laps = len(self.data)
        strategy = np.full(laps, pace_level)
        for lap in pit_laps:
            if 0 <= lap < laps:
                strategy[lap] = 0.10
        return strategy

    def compare(
        self, 
        one_stop_lap: int, 
        two_stop_lap1: int, 
        two_stop_lap2: int, 
        trials: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulates and compares the two strategies.
        """
        # 1-Stop Conservative (pace = 0.92)
        strat_a = self.generate_strategy_profiles([one_stop_lap], pace_level=0.92)
        
        # 2-Stop Aggressive (pace = 1.08)
        strat_b = self.generate_strategy_profiles([two_stop_lap1, two_stop_lap2], pace_level=1.08)
        
        # Run Monte Carlo
        times_a, pos_a = self.simulator.run_simulation(strat_a, self.sc_probs, trials=trials)
        times_b, pos_b = self.simulator.run_simulation(strat_b, self.sc_probs, trials=trials)
        
        metrics_a = self.simulator.calculate_risk_metrics(times_a, pos_a)
        metrics_b = self.simulator.calculate_risk_metrics(times_b, pos_b)
        
        # Determine Recommendation
        time_diff = metrics_a['mean'] - metrics_b['mean']
        risk_diff = metrics_a['std_dev'] - metrics_b['std_dev']
        
        if metrics_b['mean'] < metrics_a['mean'] - 2.0:
            rec_text = (
                f"AI Recommendation: Select Strategy B (2-Stop). The 2-Stop profile is expected to be "
                f"faster by {time_diff:.2f}s due to fresher tyres, overriding the extra pitstop penalty."
            )
            rec_code = "2_STOP"
        elif metrics_a['std_dev'] < metrics_b['std_dev'] - 1.5:
            rec_text = (
                f"AI Recommendation: Select Strategy A (1-Stop). The 1-Stop profile provides defensive "
                f"stability with lower volatility (Standard Deviation is {abs(risk_diff):.2f}s lower)."
            )
            rec_code = "1_STOP"
        else:
            rec_text = (
                f"AI Recommendation: Neutral. The expected time gap is marginal ({abs(time_diff):.2f}s). "
                f"Select Strategy A (1-Stop) to preserve track position."
            )
            rec_code = "NEUTRAL"
            
        return {
            'A': metrics_a,
            'B': metrics_b,
            'recommendation_text': rec_text,
            'recommendation_code': rec_code,
            'times_a': times_a,
            'times_b': times_b
        }
