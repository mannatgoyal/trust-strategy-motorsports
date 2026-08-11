import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.monte_carlo import F1MonteCarloSimulator

class F1StrategyComparisonEngine:
    """
    F1 Strategy Comparison Engine.
    Compares strategy profiles by executing joint Monte Carlo simulation runs,
    evaluating expected race times, volatility risk, and outputting tactical recommendations.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        track_id: str = "silverstone",
        track_config: Any = None,
        sc_probs: np.ndarray = None,
        confidence: np.ndarray = None
    ):
        self.data = data.copy()
        
        # Support old signature: F1StrategyComparisonEngine(data, sc_probs)
        # where the second argument is sc_probs
        actual_sc_probs = sc_probs
        actual_track_id = track_id
        if isinstance(track_id, np.ndarray):
            actual_sc_probs = track_id
            actual_track_id = "silverstone"
            
        self.confidence = confidence if confidence is not None else (
            data['StrategyConfidence'].to_numpy() if 'StrategyConfidence' in data.columns else (
                data['Trust'].to_numpy() if 'Trust' in data.columns else np.full(len(data), 0.8)
            )
        )
        
        self.simulator = F1MonteCarloSimulator(
            data=self.data,
            track_id=actual_track_id,
            track_config=track_config,
            confidence=self.confidence,
            sc_probs=actual_sc_probs
        )
        
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

    def evaluate_strategy(
        self,
        strategy_name: str,
        pit_laps: list,
        pace_level: float = 1.0,
        trials: int = 1000,
        strategy_profile: np.ndarray = None
    ) -> Any:
        """
        Evaluates a candidate strategy profile using the Monte Carlo simulator.
        Returns a StrategyDecision object.
        """
        from src.strategy_engine import StrategyDecision
        
        if strategy_profile is None:
            strategy_profile = self.generate_strategy_profiles(pit_laps, pace_level)
            
        times, positions = self.simulator.run_simulation(strategy_profile, trials=trials)
        metrics = self.simulator.calculate_risk_metrics(times, positions)
        
        # Estimate overall confidence weighted by non-pit laps
        non_pit_mask = strategy_profile > 0.10
        mean_confidence = float(np.mean(self.confidence[non_pit_mask])) if np.any(non_pit_mask) else 0.8
        
        decision = StrategyDecision(
            strategy_name=strategy_name,
            pit_laps=pit_laps,
            pace_profile=strategy_profile,
            expected_time=metrics['mean'],
            time_std=metrics['std_dev'],
            win_probability=metrics['win_probability'],
            podium_probability=metrics['podium_probability'],
            confidence=mean_confidence,
            rationale=""
        )
        # Store raw times dynamically for plotting
        decision.raw_times = times
        return decision

    def compare(
        self, 
        one_stop_lap: int, 
        two_stop_lap1: int, 
        two_stop_lap2: int, 
        trials: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulates and compares two strategies using evaluate_strategy.
        """
        # 1-Stop Conservative (pace = 0.92)
        decision_a = self.evaluate_strategy(
            strategy_name="Strategy A (1-Stop)",
            pit_laps=[one_stop_lap],
            pace_level=0.92,
            trials=trials
        )
        
        # 2-Stop Aggressive (pace = 1.08)
        decision_b = self.evaluate_strategy(
            strategy_name="Strategy B (2-Stop)",
            pit_laps=[two_stop_lap1, two_stop_lap2],
            pace_level=1.08,
            trials=trials
        )
        
        # Determine Recommendation
        time_diff = decision_a.expected_time - decision_b.expected_time
        risk_diff = decision_a.time_std - decision_b.time_std
        
        if decision_b.expected_time < decision_a.expected_time - 2.0:
            rec_text = (
                f"AI Recommendation: Select Strategy B (2-Stop). The 2-Stop profile is expected to be "
                f"faster by {time_diff:.2f}s due to fresher tyres, overriding the extra pitstop penalty."
            )
            rec_code = "2_STOP"
        elif decision_a.time_std < decision_b.time_std - 1.5:
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
            
        metrics_a = {
            'mean': decision_a.expected_time,
            'std_dev': decision_a.time_std,
            'podium_probability': decision_a.podium_probability
        }
        metrics_b = {
            'mean': decision_b.expected_time,
            'std_dev': decision_b.time_std,
            'podium_probability': decision_b.podium_probability
        }
        
        return {
            'A': metrics_a,
            'B': metrics_b,
            'recommendation_text': rec_text,
            'recommendation_code': rec_code,
            'times_a': decision_a.raw_times,
            'times_b': decision_b.raw_times,
            'decision_a': decision_a,
            'decision_b': decision_b
        }
