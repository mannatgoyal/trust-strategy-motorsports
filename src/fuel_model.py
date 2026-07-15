import numpy as np
from src.config import CONFIG

class FuelModel:
    """
    Non-linear fuel weight and consumption model.
    Accounts for diminishing weight timing benefits, track aerodynamic 
    sensitivity, and pace-dependent fuel burn coefficients.
    """
    def __init__(self, total_laps: int, initial_fuel: float = None):
        self.total_laps = total_laps
        self.initial_fuel = initial_fuel if initial_fuel is not None else CONFIG.fuel.fuel_capacity
        self.reset()
        
    def reset(self):
        """Resets the fuel tank to initial capacity"""
        self.remaining_fuel = self.initial_fuel
        
    def calculate_lap_burn(self, push_level: float) -> float:
        """
        Computes the fuel mass burned on a lap based on driver push.
        """
        # Average burn rate per lap needed to complete the race
        base_burn_rate = self.initial_fuel / max(self.total_laps, 1)
        
        # Burn rate scales with push level (aggressive throttle uses up to 15% more fuel)
        actual_burn = base_burn_rate * (1.0 + 0.15 * (push_level - 1.0))
        return float(np.clip(actual_burn, 0.5, 4.0)) # bounded F1 burn rates
        
    def step_lap(self, push_level: float) -> float:
        """
        Burns fuel for the current lap and updates remaining tank mass.
        """
        burn = self.calculate_lap_burn(push_level)
        self.remaining_fuel = float(max(0.0, self.remaining_fuel - burn))
        return self.remaining_fuel
        
    def calculate_lap_time_effect(self, remaining_fuel: float, track_mult: float = 1.0) -> float:
        """
        Calculates the non-linear timing penalty (seconds) due to fuel weight.
        Uses quadratic decay modeling diminishing returns.
        """
        # Load constants from CONFIG
        c_linear = CONFIG.fuel.fuel_penalty_linear
        c_quad = CONFIG.fuel.fuel_penalty_quadratic
        c_aero = CONFIG.fuel.aero_sensitivity
        
        # Diminishing weight timing penalty: linear coefficient + quadratic weight curve
        penalty = (c_linear * remaining_fuel) + (c_quad * (remaining_fuel ** 2))
        
        # Aerodynamic loss modifier due to higher ride height at full weight
        aero_loss = c_aero * remaining_fuel
        
        return float((penalty + aero_loss) * track_mult)
