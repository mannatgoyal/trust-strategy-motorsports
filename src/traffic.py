import numpy as np
from src.config import CONFIG

class F1TrafficSimulator:
    """
    Dynamic traffic congestion and overtaking probability simulator.
    Models aerodynamic dirty air wake loss, overtaking probability grids,
    DRS assist zones, and backmarker blue flag offsets.
    """
    def __init__(self):
        self.cfg = CONFIG.traffic
        
    def calculate_dirty_air_penalty(self, exit_gap: float, drs_active: bool = False) -> float:
        """
        Calculates time loss (seconds) due to aerodynamic turbulence in dirty air.
        """
        threshold = self.cfg.dirty_air_threshold
        if exit_gap >= threshold:
            return 0.0
            
        # Drag wake loss scales linearly inside the dirty air window
        base_loss = self.cfg.dirty_air_loss_max * ((threshold - exit_gap) / threshold)
        
        # DRS wing open mitigates wake drag losses by 30%
        drs_reduction = 0.7 if drs_active else 1.0
        return float(base_loss * drs_reduction)
        
    def calculate_overtake_probability(
        self, 
        grip_self: float, 
        grip_ahead: float, 
        gap: float, 
        drs_zone: bool, 
        closing_speed: float
    ) -> float:
        """
        Calculates the probability of executing a clean overtake on a given lap.
        Uses a log-sigmoid logistic activation function.
        """
        if gap > 2.0:
            return 0.0 # too far to attempt overtake
            
        # 1. Grip offset advantage
        grip_delta = grip_self - grip_ahead
        
        # 2. DRS assist indicator
        drs_factor = 1.5 if (drs_zone and gap < 1.0) else 0.0
        
        # Log-sigmoid logit input
        logit = (self.cfg.sigmoid_overtake_scale * (
            3.0 * grip_delta 
            + drs_factor 
            + 2.0 * closing_speed
            - self.cfg.base_defensive_factor
        ))
        
        # Logistic sigmoid activation
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(prob, 0.0, 1.0))
        
    def estimate_traffic_time_loss(
        self, 
        exit_gap: float, 
        grip_self: float, 
        grip_ahead: float, 
        drs_zone: bool,
        closing_speed: float,
        is_backmarker: bool = False
    ) -> float:
        """
        Returns the expected probabilistic time loss (seconds) due to traffic bottlenecks.
        """
        # Blue flag rule: backmarkers must let lead cars pass with minimal (0.2s) delay
        if is_backmarker and exit_gap < 1.0:
            return 0.2
            
        dirty_air = self.calculate_dirty_air_penalty(exit_gap, drs_active=(drs_zone and exit_gap < 1.0))
        
        if exit_gap < 1.5:
            p_overtake = self.calculate_overtake_probability(grip_self, grip_ahead, exit_gap, drs_zone, closing_speed)
            # If overtaking is failed, we are stuck behind (incurs pacing penalty)
            fight_loss = (1.0 - p_overtake) * 1.8
            return float(dirty_air + fight_loss)
            
        return float(dirty_air)
