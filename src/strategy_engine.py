from dataclasses import dataclass, field
import numpy as np
from typing import List

@dataclass
class StrategyDecision:
    strategy_name: str
    pit_laps: List[int]
    pace_profile: np.ndarray
    expected_time: float = 0.0
    time_std: float = 0.0
    win_probability: float = 0.0
    podium_probability: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    rationale: str = ""

class StrategyDecisionPolicy:
    """
    Decision-theoretic policy to score and rank F1 strategies based on 
    expected stint duration, strategic volatility (risk), and a risk-aversion coefficient.
    """
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_acceptable_time_loss: float = 2.0,
        max_risk_increase: float = 1.5
    ):
        self.risk_aversion = risk_aversion
        self.max_acceptable_time_loss = max_acceptable_time_loss
        self.max_risk_increase = max_risk_increase
        
    def evaluate(self, decision: StrategyDecision) -> float:
        """
        Calculates the decision-theoretic utility score.
        U = -E[T] - lambda * sigma_T
        """
        score = -decision.expected_time - self.risk_aversion * decision.time_std
        decision.score = score
        return score
        
    def select_best(self, candidates: List[StrategyDecision]) -> StrategyDecision:
        """
        Selects the best strategy from candidate options by evaluating utility scores.
        """
        if not candidates:
            raise ValueError("Candidate list is empty.")
            
        for cand in candidates:
            self.evaluate(cand)
            
        # Sort by score descending (highest score/least negative utility)
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return sorted_candidates[0]

def recommend_pit(wear: float, sc_threat: float) -> bool:
    """
    Unified decision rule to determine if a pit stop is recommended
    based on tire wear and safety car threat levels.
    """
    return wear > 0.60 or (sc_threat > 0.35 and wear > 0.40)
