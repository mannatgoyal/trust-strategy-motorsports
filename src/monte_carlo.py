import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.config import CONFIG
from src.tire_degradation import TireDegradationModel
from src.fuel_model import FuelModel
from src.pit_stop import PitStopSimulator
from src.traffic import F1TrafficSimulator
from src.safety_car import BayesianSafetyCarModel
from src.weather import DynamicWeatherSystem

class F1MonteCarloSimulator:
    """
    State-of-the-art Monte Carlo simulator coordinating all thermodynamic tyre wear, 
    fuel timing, Bayesian Safety Cars, dynamic weather, and traffic wake bottlenecks.
    
    Joint Probability Sampling:
      Weather Dampness -> Driver Pacing Volatility & Overtaking Incidents -> Safety Car Posteriors.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.laps = len(data)
        self.base_trust = data['Trust'].values if 'Trust' in data.columns else np.full(self.laps, 0.8)
        
    def simulate_trial(self, strategy: np.ndarray, sc_probs: np.ndarray) -> Tuple[float, int]:
        """
        Simulates a single stint timeline, sampling all random elements jointly.
        """
        tyre_model = TireDegradationModel(compound='medium')
        fuel_model = FuelModel(total_laps=self.laps)
        pit_model = PitStopSimulator()
        traffic_model = F1TrafficSimulator()
        sc_model = BayesianSafetyCarModel()
        weather_model = DynamicWeatherSystem(rain_probability=0.12)
        
        elapsed_time = 0.0
        pit_lap = int(np.argmin(strategy))
        current_position = int(self.data.loc[0, 'Position']) if 'Position' in self.data.columns else 3
        
        for k in range(self.laps):
            push_level = float(strategy[k])
            
            # A. Joint Weather state update
            dampness, weather_state = weather_model.step_lap(k)
            wear_mult = weather_model.get_tyre_wear_multiplier(tyre_model.compound, weather_state)
            grip_mult = weather_model.get_track_grip_multiplier(tyre_model.compound, weather_state)
            
            # B. Correlated parameters: rain increases driver volatility & crash incidents
            if weather_state == 'Wet':
                pacing_std = CONFIG.monte_carlo.pacing_std * 2.0
                incident_prob = 0.08
            elif weather_state == 'Damp':
                pacing_std = CONFIG.monte_carlo.pacing_std * 1.4
                incident_prob = 0.04
            else:
                pacing_std = CONFIG.monte_carlo.pacing_std
                incident_prob = 0.015
                
            recent_incidents = 1 if np.random.random() < incident_prob else 0
            
            # C. Bayesian Safety Car drawing (posterior updates dynamic incidents)
            sc_est = sc_model.estimate_posterior_probabilities(k+1, "Silverstone", weather_state, recent_incidents)
            sc_active = np.random.random() < sc_est['Combined']
            
            # Pacing consistency noise (driver error volatility)
            pacing_error = np.random.normal(0.0, pacing_std)
            
            if k == pit_lap:
                # Pit Stop Simulation
                stop_results = pit_model.simulate_stop(track_name="Silverstone")
                pit_time = stop_results['total_loss']
                if sc_active:
                    # Pit timing benefit under safety car (slower delta loss)
                    pit_time -= 10.0
                elapsed_time += pit_time + stop_results['cold_tire_penalty']
                
                # Traffic wake exit gap evaluation
                col_gap = 'ExitGap_SC' if sc_active else 'ExitGap'
                exit_gap = self.data.loc[k, col_gap] if col_gap in self.data.columns else 4.0
                traffic_loss = traffic_model.estimate_traffic_time_loss(
                    exit_gap=exit_gap,
                    grip_self=0.9,
                    grip_ahead=0.7,
                    drs_zone=(k % 5 == 0),
                    closing_speed=0.5
                )
                elapsed_time += traffic_loss
                tyre_model.reset()
            else:
                # Standard pacing lap
                wear, temp, grip = tyre_model.step_lap(push_level, 35.0, 25.0)
                grip = grip * grip_mult
                
                # Wear and temperature pacing penalties
                wear_loss = tyre_model.calculate_lap_penalty(grip) * wear_mult
                
                # Fuel timing corrections
                fuel_mass = fuel_model.step_lap(push_level)
                fuel_loss = fuel_model.calculate_lap_time_effect(fuel_mass)
                
                # Aerodynamic wake penalty
                gap_ahead = self.data.loc[k, 'GapAhead'] if 'GapAhead' in self.data.columns else 5.0
                dirty_air_loss = traffic_model.calculate_dirty_air_penalty(gap_ahead, drs_active=(gap_ahead < 1.0))
                
                lap_time = 90.0 - 5.0 * self.base_trust[k] + wear_loss + fuel_loss + dirty_air_loss + pacing_error
                
                # Safety car track speeds
                if sc_active:
                    lap_time += np.random.uniform(15.0, 25.0)
                    
                # Sigmoidal overtaking updates position
                if gap_ahead < 1.0:
                    p_overtake = traffic_model.calculate_overtake_probability(
                        grip_self=grip,
                        grip_ahead=0.75,
                        gap=gap_ahead,
                        drs_zone=(k % 5 == 0),
                        closing_speed=0.2
                    )
                    if np.random.random() < p_overtake:
                        current_position = max(1, current_position - 1)
                        
                elapsed_time += lap_time
                
        return elapsed_time, current_position

    def run_simulation(self, strategy: np.ndarray, sc_probs: np.ndarray, trials: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs M Monte Carlo stint trials.
        """
        times = np.zeros(trials)
        positions = np.zeros(trials, dtype=int)
        for i in range(trials):
            t, pos = self.simulate_trial(strategy, sc_probs)
            times[i] = t
            positions[i] = pos
        return times, positions

    def calculate_risk_metrics(self, times: np.ndarray, positions: np.ndarray) -> Dict[str, Any]:
        """
        Calculates expected distributions, confidence intervals, and strategic risk bounds.
        """
        mean_time = float(np.mean(times))
        std_dev = float(np.std(times))
        var_95 = float(np.percentile(times, 95))
        ci_lower = float(np.percentile(times, 2.5))
        ci_upper = float(np.percentile(times, 97.5))
        
        # Overtaking / finishing outcomes
        expected_pos = float(np.mean(positions))
        win_prob = float(np.mean(positions == 1))
        podium_prob = float(np.mean(positions <= 3))
        
        return {
            'mean': mean_time,
            'std_dev': std_dev,
            'var_95': var_95,
            'ci_95': (ci_lower, ci_upper),
            'expected_position': expected_pos,
            'win_probability': win_prob,
            'podium_probability': podium_prob
        }
