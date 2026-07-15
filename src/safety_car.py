import numpy as np
from src.config import CONFIG
from typing import Dict

class BayesianSafetyCarModel:
    """
    Bayesian Safety Car risk estimator.
    Estimates P(SC | lap, circuit, weather, incidents) for VSC, Safety Car, 
    and Red Flags by combining prior odds and likelihood factors.
    """
    def __init__(self):
        self.cfg = CONFIG.safety_car
        
    def get_circuit_prior_multiplier(self, track_name: str) -> float:
        """
        Returns track-specific prior risk multipliers.
        """
        if track_name and any(alias in track_name for alias in ["Silverstone", "British"]):
            return 1.3 # Silverstone has higher risk profile
        elif track_name and any(alias in track_name for alias in ["Yas Marina", "Abu Dhabi"]):
            return 0.9 # Yas Marina has modern wide runoffs
        return 1.0

    def estimate_posterior_probabilities(
        self, 
        lap_num: int, 
        track_name: str, 
        weather_state: str, 
        recent_incidents: int
    ) -> Dict[str, float]:
        """
        Calculates the Bayesian posterior probability of VSC, SC, and Red Flags.
        """
        circuit_mult = self.get_circuit_prior_multiplier(track_name)
        
        # 1. Base prior values
        p_vsc_prior = self.cfg.vsc_prior * circuit_mult
        p_sc_prior = self.cfg.sc_prior * circuit_mult
        p_rf_prior = self.cfg.red_flag_prior * circuit_mult
        
        # Likelihood ratio modifiers
        # A. Lap modifier (first lap grid bunching risk multiplier = 3.0)
        lap_likelihood = 3.0 if lap_num == 1 else 1.0
        
        # B. Weather modifier (rain multiplier = 2.5)
        weather_likelihood = self.cfg.rain_multiplier if weather_state.lower() == 'wet' else 1.0
        
        # C. Incident multiplier (escalates with track crashes)
        incident_likelihood = 1.0 + 2.5 * recent_incidents
        
        # Net likelihood multiplier
        total_likelihood = lap_likelihood * weather_likelihood * incident_likelihood
        
        # Bayesian update using odds formulation: posterior odds = prior odds * likelihood
        def update_odds(prior_p: float, likelihood: float) -> float:
            prior_p = min(0.999, max(0.001, prior_p))
            prior_odds = prior_p / (1.0 - prior_p)
            posterior_odds = prior_odds * likelihood
            posterior_p = posterior_odds / (1.0 + posterior_odds)
            return float(np.clip(posterior_p, 0.0, 0.95))
            
        p_vsc = update_odds(p_vsc_prior, total_likelihood)
        p_sc = update_odds(p_sc_prior, total_likelihood)
        p_rf = update_odds(p_rf_prior, total_likelihood)
        
        return {
            'VSC': p_vsc,
            'SC': p_sc,
            'RedFlag': p_rf,
            'Combined': float(np.clip(p_vsc + p_sc + p_rf, 0.0, 0.98))
        }
