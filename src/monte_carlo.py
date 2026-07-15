import numpy as np

class F1MonteCarloSimulator:
    """
    Monte Carlo race strategy simulator for risk and variance analysis.
    Simulates random Safety Car occurrences to generate probability distributions
    of total stint completion times.
    """
    def __init__(self, data, traffic_penalty=1.5):
        self.data = data.copy()
        self.laps = len(data)
        self.base_trust = data['Trust'].values
        self.traffic_penalty = traffic_penalty
        
    def simulate_stint(self, strategy, sc_probs):
        """
        Simulates a single stint timeline with randomized Safety Car draws.
        """
        elapsed_time = 0.0
        tire_wear = 0.0
        pit_lap = int(np.argmin(strategy))
        
        for k in range(self.laps):
            # Draw random safety car event
            sc_active = np.random.random() < sc_probs[k]
            
            if k == pit_lap:
                # Pit stop overhead
                pit_loss = 12.0 if sc_active else 22.0
                elapsed_time += pit_loss
                
                # Check traffic congestion release penalty
                col_gap = 'ExitGap_SC' if sc_active else 'ExitGap'
                if col_gap in self.data.columns:
                    exit_gap = self.data.loc[k, col_gap]
                    if exit_gap < 1.5:
                        elapsed_time += self.traffic_penalty
                
                # Reset tire wear
                tire_wear = 0.0
            else:
                # Normal running lap time model
                lap_time = 90.0 - 5.0 * self.base_trust[k] + 4.0 * tire_wear
                
                # If safety car is active, speed is restricted (adds pace overhead)
                if sc_active:
                    lap_time += np.random.uniform(15.0, 25.0)
                    
                elapsed_time += lap_time
                
                # Degrade tire wear
                tire_wear += 0.01 * (strategy[k] ** 2)
                
        return elapsed_time

    def run_simulation(self, strategy, sc_probs, trials=1000):
        """
        Runs M simulated stint trials for a given strategy.
        """
        stint_times = np.zeros(trials)
        for i in range(trials):
            stint_times[i] = self.simulate_stint(strategy, sc_probs)
        return stint_times

    def calculate_risk_metrics(self, stint_times):
        """
        Computes statistical distribution and risk metrics.
        """
        mean_val = float(np.mean(stint_times))
        std_val = float(np.std(stint_times))
        var_95 = float(np.percentile(stint_times, 95))
        return {
            'mean': mean_val,
            'std_dev': std_val,
            'var_95': var_95
        }
