import numpy as np
from typing import Tuple
from src.config import CONFIG

class TireDegradationModel:
    """
    Advanced thermodynamic tire degradation model.
    Simulates tire temperature transitions, compound wear accumulation, 
    instantaneous grip, and the lap time cliff effect.
    """
    def __init__(self, compound: str = 'Medium'):
        self.compound = compound.lower()
        
        # Load compound configuration parameters from YAML CONFIG
        if self.compound == 'soft':
            self.cfg = CONFIG.tyre.soft
        elif self.compound == 'hard':
            self.cfg = CONFIG.tyre.hard
        else:
            self.compound = 'medium'
            self.cfg = CONFIG.tyre.medium
            
        self.reset()
        
    def reset(self):
        """Resets tire state to brand new conditions"""
        self.wear = 0.0
        # Initialize tire temperature at Track Temperature baseline + 10C
        self.temperature = 35.0 + 10.0
        
    def step_lap(self, push_level: float, track_temp: float, ambient_temp: float) -> Tuple[float, float, float]:
        """
        Executes one lap simulation step.
        Returns:
            (wear_percent, temperature, grip_coefficient)
        """
        # 1. Thermal dynamics: friction heating vs track/air cooling
        base_friction_heat = CONFIG.tyre.friction_heat
        cooling_rate = CONFIG.tyre.cooling_rate
        
        # Friction heating is quadratic with driver throttle/push level
        heat_gain = base_friction_heat * (push_level ** 2) * self.cfg.base_grip
        
        # Heat cooling scales with track/ambient temp difference
        heat_loss = cooling_rate * (self.temperature - ambient_temp)
        
        self.temperature = float(np.clip(
            self.temperature + heat_gain - heat_loss,
            track_temp,
            140.0 # Maximum physical tyre temp limit
        ))
        
        # 2. Thermal window evaluation (Warm-up vs. Optimal vs. Overheating)
        if self.temperature < self.cfg.thermal_window_min:
            # Warm-up phase penalty
            thermal_factor = 1.0 - 0.10 * ((self.cfg.thermal_window_min - self.temperature) / self.cfg.thermal_window_min) ** 2
        elif self.temperature > self.cfg.thermal_window_max:
            # Overheating degradation penalty
            thermal_factor = 1.0 - 0.15 * ((self.temperature - self.cfg.thermal_window_max) / self.cfg.thermal_window_max) ** 2
        else:
            thermal_factor = 1.0
            
        # 3. Wear accumulation (wear rate escalates with high temperatures)
        temp_wear_modifier = 1.0 + max(0.0, self.temperature - self.cfg.thermal_window_max) * 0.015
        wear_increment = self.cfg.wear_rate * (push_level ** 1.5) * temp_wear_modifier
        self.wear = float(np.clip(self.wear + wear_increment, 0.0, 1.0))
        
        # 4. Grip coefficient & Cliff effect solver
        grip_wear_factor = 1.0 - 0.35 * self.wear
        
        # Exponential cliff penalty if wear threshold exceeded
        cliff_penalty = 0.0
        if self.wear >= self.cfg.cliff_wear:
            cliff_penalty = 0.4 * np.exp(6.0 * (self.wear - self.cfg.cliff_wear))
            
        grip = float(np.clip(
            self.cfg.base_grip * thermal_factor * grip_wear_factor - cliff_penalty,
            0.1, # Min physical grip boundary
            1.2
        ))
        
        return self.wear, self.temperature, grip
        
    def calculate_lap_penalty(self, grip: float) -> float:
        """
        Calculates the lap time loss (seconds) based on current tire grip.
        """
        # Loss scales up to 5.0 seconds at minimal grip (0.1)
        return float(5.0 * (1.05 - grip))
