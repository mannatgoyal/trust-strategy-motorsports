import numpy as np
from typing import Dict, Tuple
from src.config import CONFIG

class PitStopSimulator:
    """
    Detailed Pit Stop simulation handler.
    Partitions the pit stop duration into entry transit, stationary service,
    exit transit, and cold tire grip timing penalties, scaling properties 
    depending on track constraints.
    """
    def __init__(self):
        self.cfg = CONFIG.pit
        
    def get_track_scaling(self, track_name: str) -> Tuple[float, float]:
        """
        Returns multipliers for pit lane entry and exit transits.
        """
        # Silverstone has a longer pit lane entry/exit loop
        if track_name and any(alias in track_name for alias in ["Silverstone", "British"]):
            return 1.2, 1.2
        # Yas Marina (Abu Dhabi) features a unique slow underpass exit tunnel
        elif track_name and any(alias in track_name for alias in ["Yas Marina", "Abu Dhabi"]):
            return 1.0, 1.4
        return 1.0, 1.0

    def simulate_stop(self, track_name: str, random_seed: int = None) -> Dict[str, float]:
        """
        Simulates the elapsed time components of a pit stop.
        Returns:
            Dict containing entry, stationary, exit, out-lap, and total time.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        entry_mult, exit_mult = self.get_track_scaling(track_name)
        
        # 1. Entry and Exit transit times
        entry_time = self.cfg.pit_entry_loss * entry_mult
        exit_time = self.cfg.pit_exit_loss * exit_mult
        
        # 2. Stationary service (wheel gun) duration with normal variance
        stationary_time = np.random.normal(self.cfg.stationary_mean, self.cfg.stationary_std)
        # Prevent physical impossibilities (pit stop under 1.5 seconds)
        stationary_time = max(1.5, stationary_time)
        
        # 3. Cold tire out-lap pace penalty
        cold_tire_penalty = self.cfg.cold_tire_outlap_loss
        
        total_pit_loss = entry_time + stationary_time + exit_time
        
        return {
            'entry': float(entry_time),
            'stationary': float(stationary_time),
            'exit': float(exit_time),
            'cold_tire_penalty': float(cold_tire_penalty),
            'total_loss': float(total_pit_loss)
        }
