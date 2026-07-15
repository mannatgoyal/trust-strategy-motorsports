import numpy as np
import pandas as pd
from typing import Tuple, Dict

class GameTheoryStrategist:
    """
    Game theory strategic optimizer.
    Computes Nash equilibrium, Stackelberg leadership, and mixed-strategy 
    probabilities for motorsport strategic options.
    """
    def __init__(self, data: pd.DataFrame, traffic_penalty: float = 0.15):
        self.data = data.copy()
        self.laps = len(data)
        self.base_confidence = data['Trust'].values if 'Trust' in data.columns else np.full(self.laps, 0.8)
        self.traffic_penalty = traffic_penalty

    def calculate_utility(self, strategy_self: np.ndarray, strategy_comp: np.ndarray, sc_active: bool = False) -> float:
        """
        Computes stint utility incorporating pacing, degradation wear, and dirty air traffic penalties.
        """
        # Pace score (mean confidence weighted by push strategy)
        pace_score = np.mean(strategy_self * self.base_confidence)
        
        # Degradation penalty (quadratic wear)
        wear = np.zeros(self.laps)
        current_wear = 0.0
        for k in range(self.laps):
            current_wear += 0.012 * (strategy_self[k] ** 2)
            wear[k] = current_wear
        deg_penalty = np.mean(wear)
        
        # Track position (DRS zone relative push timing)
        relative_push = strategy_self - strategy_comp
        drs_laps = np.arange(5, self.laps, 5)
        position_factor = np.mean(relative_push)
        if len(drs_laps) > 0:
            position_factor += 0.25 * np.mean(relative_push[drs_laps])
            
        utility = 0.50 * pace_score - 0.15 * deg_penalty + 0.35 * position_factor
        
        # Pit stop exit gap dirty air penalty
        pit_lap = int(np.argmin(strategy_self))
        col_gap = 'ExitGap_SC' if sc_active else 'ExitGap'
        
        if col_gap in self.data.columns and 0 <= pit_lap < len(self.data):
            exit_gap = self.data.loc[pit_lap, col_gap]
            if exit_gap < 1.5:
                utility -= self.traffic_penalty
                
        return float(np.clip(utility, 0.0, 1.0))

    def get_payoff_matrix(self, strat_a_con: np.ndarray, strat_a_agg: np.ndarray, strat_b_con: np.ndarray, strat_b_agg: np.ndarray, sc_active: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 2x2 payoff matrices for Leader (Driver A) and Follower (Driver B).
        Row: Driver A (Con vs Agg), Col: Driver B (Con vs Agg).
        """
        payoff_a = np.zeros((2, 2))
        payoff_b = np.zeros((2, 2))
        
        # Row 0, Col 0: Con vs Con
        payoff_a[0, 0] = self.calculate_utility(strat_a_con, strat_b_con, sc_active)
        payoff_b[0, 0] = self.calculate_utility(strat_b_con, strat_a_con, sc_active)
        
        # Row 0, Col 1: Con vs Agg
        payoff_a[0, 1] = self.calculate_utility(strat_a_con, strat_b_agg, sc_active)
        payoff_b[0, 1] = self.calculate_utility(strat_b_agg, strat_a_con, sc_active)
        
        # Row 1, Col 0: Agg vs Con
        payoff_a[1, 0] = self.calculate_utility(strat_a_agg, strat_b_con, sc_active)
        payoff_b[1, 0] = self.calculate_utility(strat_b_con, strat_a_agg, sc_active)
        
        # Row 1, Col 1: Agg vs Agg
        payoff_a[1, 1] = self.calculate_utility(strat_a_agg, strat_b_agg, sc_active)
        payoff_b[1, 1] = self.calculate_utility(strat_b_agg, strat_a_agg, sc_active)
        
        return payoff_a, payoff_b

    def solve_stackelberg(self, payoff_a: np.ndarray, payoff_b: np.ndarray) -> Tuple[int, int]:
        """
        Solves for the Stackelberg Leader-Follower equilibrium.
        Driver A is the Leader (Row select), Driver B is the Follower (Col select).
        """
        # B's best response for Row 0
        br_col_for_row0 = int(np.argmax(payoff_b[0, :]))
        # B's best response for Row 1
        br_col_for_row1 = int(np.argmax(payoff_b[1, :]))
        
        # A evaluates payoffs under B's best responses
        payoff_a_row0 = payoff_a[0, br_col_for_row0]
        payoff_a_row1 = payoff_a[1, br_col_for_row1]
        
        if payoff_a_row0 >= payoff_a_row1:
            return 0, br_col_for_row0
        return 1, br_col_for_row1

    def solve_mixed_nash(self, payoff_a: np.ndarray, payoff_b: np.ndarray) -> Tuple[float, float]:
        """
        Calculates mixed-strategy Nash equilibrium probabilities of playing Conservative.
        Returns:
            (p_a_conservative, p_b_conservative)
        """
        # Indifference calculation for Driver A (row player)
        denom_a = (payoff_b[0, 0] - payoff_b[0, 1] - payoff_b[1, 0] + payoff_b[1, 1])
        if abs(denom_a) < 1e-4:
            q_b = 0.5 # default if payoffs are equal
        else:
            q_b = (payoff_b[1, 1] - payoff_b[0, 1]) / denom_a
            
        # Indifference calculation for Driver B (column player)
        denom_b = (payoff_a[0, 0] - payoff_a[0, 1] - payoff_a[1, 0] + payoff_a[1, 1])
        if abs(denom_b) < 1e-4:
            p_a = 0.5
        else:
            p_a = (payoff_a[1, 1] - payoff_a[1, 0]) / denom_b
            
        return float(np.clip(p_a, 0.0, 1.0)), float(np.clip(q_b, 0.0, 1.0))
        
    def calculate_sc_probability(self, track_name: str) -> np.ndarray:
        """
        Track baseline safety car probabilities (returns array of shape self.laps).
        """
        p_base = 0.015 if track_name and "Silverstone" in track_name else 0.010
        sc_probs = np.full(self.laps, p_base)
        if self.laps > 0:
            sc_probs[0] = min(p_base * 3.0, 1.0)
        return sc_probs
        
    def calculate_expected_payoff(self, strat_a: np.ndarray, strat_b: np.ndarray, sc_probs: np.ndarray) -> Tuple[float, float]:
        """
        Expected payoff incorporating Safety Car transitions.
        """
        pit_a = int(np.argmin(strat_a))
        p_sc_a = sc_probs[pit_a] if 0 <= pit_a < len(sc_probs) else 0.0
        ut_a_gf = self.calculate_utility(strat_a, strat_b, sc_active=False)
        ut_a_sc = self.calculate_utility(strat_a, strat_b, sc_active=True)
        exp_a = (1.0 - p_sc_a) * ut_a_gf + p_sc_a * ut_a_sc
        
        pit_b = int(np.argmin(strat_b))
        p_sc_b = sc_probs[pit_b] if 0 <= pit_b < len(sc_probs) else 0.0
        ut_b_gf = self.calculate_utility(strat_b, strat_a, sc_active=False)
        ut_b_sc = self.calculate_utility(strat_b, strat_a, sc_active=True)
        exp_b = (1.0 - p_sc_b) * ut_b_gf + p_sc_b * ut_b_sc
        
        return float(exp_a), float(exp_b)
