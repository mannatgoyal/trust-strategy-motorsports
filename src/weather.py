import numpy as np
from typing import Tuple

class DynamicWeatherSystem:
    """
    Dynamic track weather simulator.
    Models track dampness levels and handles transitions between Dry, Damp, 
    and Wet conditions, altering grip coefficients and tire degradation rates.
    """
    def __init__(self, rain_probability: float = 0.10):
        self.rain_probability = rain_probability
        self.reset()
        
    def reset(self):
        """Resets the track state to bone dry conditions"""
        self.dampness = 0.0 # 0.0 is dry, 1.0 is flooded
        self.is_raining = False
        
    def step_lap(self, lap_num: int, random_seed: int = None) -> Tuple[float, str]:
        """
        Updates the track dampness and weather state.
        Returns:
            (dampness_value, weather_state)
        """
        if random_seed is not None:
            np.random.seed(random_seed + lap_num)
            
        # 1. Weather transition random walk
        if self.is_raining:
            # 20% chance of rain stopping
            if np.random.random() < 0.20:
                self.is_raining = False
        else:
            # Chance of rain starting scales with baseline probability
            if np.random.random() < self.rain_probability:
                self.is_raining = True
                
        # 2. Dampness transition
        if self.is_raining:
            # Track gets wetter
            self.dampness = min(1.0, self.dampness + np.random.uniform(0.05, 0.12))
        else:
            # Track dries out slowly
            self.dampness = max(0.0, self.dampness - np.random.uniform(0.03, 0.08))
            
        # 3. Categorize State
        if self.dampness < 0.15:
            state = 'Dry'
        elif self.dampness < 0.50:
            state = 'Damp'
        else:
            state = 'Wet'
            
        return float(self.dampness), state

    def get_tyre_wear_multiplier(self, compound: str, state: str) -> float:
        """
        Calculates wear rate modifiers based on tire-track compatibility.
        """
        compound_lower = compound.lower()
        
        # Dry slicks run on damp/wet tracks overheat and wear rapidly
        if compound_lower in ['soft', 'medium', 'hard']:
            if state == 'Damp':
                return 2.5
            elif state == 'Wet':
                return 5.0
                
        # Wet weather compounds wear out extremely quickly on dry asphalt
        elif compound_lower in ['intermediate', 'wet']:
            if state == 'Dry':
                return 4.5
                
        return 1.0

    def get_track_grip_multiplier(self, compound: str, state: str) -> float:
        """
        Calculates grip reduction factor due to standing water and tire tread pattern.
        """
        compound_lower = compound.lower()
        
        if state == 'Dry':
            if compound_lower in ['soft', 'medium', 'hard']:
                return 1.0
            else:
                return 0.85 # wet tires are slower on dry tracks
        elif state == 'Damp':
            if compound_lower == 'intermediate':
                return 0.95
            elif compound_lower in ['soft', 'medium', 'hard']:
                return 0.70 # slicks slide on damp tracks
            else:
                return 0.80
        else: # Wet track
            if compound_lower == 'wet':
                return 0.90
            elif compound_lower == 'intermediate':
                return 0.75
            else:
                return 0.40 # slicks hydroplane on wet tracks
