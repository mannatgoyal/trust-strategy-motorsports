import numpy as np
import pandas as pd

class GameTheoryStrategist:
    """
    Game theory-based race strategy optimization for F1.
    Calculates strategy options using telemetry-driven utility functions.
    Includes stochastic modeling for Safety Cars (SC).
    """
    
    def __init__(self, data, w_pace=0.50, w_position=0.35, w_deg=0.15, traffic_penalty=0.15):
        self.data = data.copy()
        self.laps = len(data)
        self.base_trust = data['Trust'].values
        
        # Weights for the utility function
        self.w_pace = w_pace
        self.w_position = w_position
        self.w_deg = w_deg
        
        # Congestion penalty for pitting in traffic
        self.traffic_penalty = traffic_penalty

    def calculate_sc_probability(self, track_name):
        """
        Computes lap-by-lap probability of a Safety Car (SC) deployment.
        """
        # Determine track base probability
        if track_name and any(alias in track_name for alias in ["Silverstone", "British"]):
            p_base = 0.015
        else:
            p_base = 0.010
            
        sc_probs = np.full(self.laps, p_base)
        
        # Incident risk spikes on Lap 1
        if self.laps > 0:
            sc_probs[0] = min(p_base * 3.0, 1.0)
            
        # Tire wear risk modifier (spikes toward the end of typical stints)
        deg_multiplier = np.linspace(0.0, 0.010, self.laps)
        sc_probs = np.clip(sc_probs + deg_multiplier, 0.0, 1.0)
        
        return sc_probs

    def calculate_utility(self, strategy_self, strategy_competitor, sc_active=False):
        """
        Computes the utility of a strategy vector S_self given S_competitor.
        U = w_pace * Pace - w_deg * Degradation + w_position * PositionFactor - CongestionPenalty
        """
        # 1. Pace: Mean push * base trust
        pace_score = np.mean(strategy_self * self.base_trust)
        
        # 2. Degradation: Cumulative tire wear based on push level
        # Aggressive push yields quadratic wear penalty
        wear = np.zeros(self.laps)
        current_wear = 0.0
        for k in range(self.laps):
            current_wear += 0.01 * (strategy_self[k] ** 2)
            wear[k] = current_wear
        deg_penalty = np.mean(wear)
        
        # 3. Track Position: Defense/overtake advantage based on relative push delta
        # Higher push on key laps (e.g. DRS zones every 5 laps) yields a positional advantage
        relative_push = strategy_self - strategy_competitor
        drs_laps = np.arange(5, self.laps, 5)
        
        # Overtaking / defending reward
        position_factor = np.mean(relative_push)
        if len(drs_laps) > 0:
            position_factor += 0.2 * np.mean(relative_push[drs_laps])
            
        # Utility formulation
        utility = (self.w_pace * pace_score 
                   - self.w_deg * deg_penalty 
                   + self.w_position * position_factor)
        
        # 4. Traffic Congestion Penalty: Check if pitting results in dirty air
        # Locate the pit lap (where the strategy has the minimum push/dip)
        pit_lap = int(np.argmin(strategy_self))
        
        # Use alternate column under Safety Car to account for lower pit lane delta loss (12s vs 22s)
        col_gap = 'ExitGap_SC' if sc_active else 'ExitGap'
        
        if col_gap in self.data.columns and 0 <= pit_lap < len(self.data):
            exit_gap = self.data.loc[pit_lap, col_gap]
            # Penalty applied if rejoining less than 1.5s behind another car
            if exit_gap < 1.5:
                utility -= self.traffic_penalty
        
        # Clip utility to standard [0, 1] range representing normalized payoff
        return float(np.clip(utility, 0, 1))

    def nash_equilibrium(self):
        """
        Computes the Nash equilibrium strategy (Conservative approach).
        Driver optimizes pace while keeping tire wear low.
        """
        strategy = self.base_trust.copy()
        
        # Conservative Pit Strategy (around 40% race distance)
        pit_lap = int(self.laps * 0.4)
        strategy[max(0, pit_lap-2):min(self.laps, pit_lap+2)] *= 0.75
        
        # Linear moderate wear factor
        wear_factor = np.linspace(1.0, 0.88, self.laps)
        strategy = strategy * wear_factor
        
        # Fuel saving in the final third
        final_stint = int(self.laps * 0.7)
        strategy[final_stint:] *= 0.95
        
        return np.clip(strategy, 0, 1)

    def stackelberg_leadership(self):
        """
        Computes the Stackelberg leader strategy (Aggressive approach).
        Driver pushes early to gain track position, accepting higher tire wear.
        """
        strategy = self.base_trust.copy()
        
        # Aggressive early push (first 25% of the race)
        early_phase = int(self.laps * 0.25)
        strategy[:early_phase] *= 1.1
        
        # Late Pit Strategy (around 50% race distance)
        pit_lap = int(self.laps * 0.5)
        strategy[max(0, pit_lap-2):min(self.laps, pit_lap+2)] *= 0.7
        
        # DRS zones push
        drs_laps = np.arange(5, self.laps, 5)
        strategy[drs_laps] = np.minimum(strategy[drs_laps] * 1.15, 1.0)
        
        # Faster linear wear factor
        wear_factor = np.linspace(1.0, 0.82, self.laps)
        strategy = strategy * wear_factor
        
        return np.clip(strategy, 0, 1)

    def calculate_payoff(self, strategy_a, strategy_b):
        """
        Calculates the payoffs for Driver A and Driver B under standard Green Flag.
        """
        payoff_a = self.calculate_utility(strategy_a, strategy_b, sc_active=False)
        payoff_b = self.calculate_utility(strategy_b, strategy_a, sc_active=False)
        return payoff_a, payoff_b

    def calculate_expected_payoff(self, strategy_a, strategy_b, sc_probs):
        """
        Calculates the expected stochastic payoffs incorporating Safety Car likelihood.
        """
        # Driver A Pit Lap & SC Probability
        pit_lap_a = int(np.argmin(strategy_a))
        p_sc_a = sc_probs[pit_lap_a] if 0 <= pit_lap_a < len(sc_probs) else 0.0
        
        ut_a_green = self.calculate_utility(strategy_a, strategy_b, sc_active=False)
        ut_a_sc = self.calculate_utility(strategy_a, strategy_b, sc_active=True)
        expected_payoff_a = (1.0 - p_sc_a) * ut_a_green + p_sc_a * ut_a_sc
        
        # Driver B Pit Lap & SC Probability
        pit_lap_b = int(np.argmin(strategy_b))
        p_sc_b = sc_probs[pit_lap_b] if 0 <= pit_lap_b < len(sc_probs) else 0.0
        
        ut_b_green = self.calculate_utility(strategy_b, strategy_a, sc_active=False)
        ut_b_sc = self.calculate_utility(strategy_b, strategy_a, sc_active=True)
        expected_payoff_b = (1.0 - p_sc_b) * ut_b_green + p_sc_b * ut_b_sc
        
        return float(expected_payoff_a), float(expected_payoff_b)
