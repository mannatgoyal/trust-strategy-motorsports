import numpy as np
from scipy.optimize import minimize

class F1TrajectoryOptimizer:
    """
    Continuous control trajectory optimization for F1 pacing.
    Solves for optimal throttle push (u) and ERS boost (b) sequences
    subject to tire degradation and battery depletion constraints.
    """
    
    def __init__(self, base_trust, regen_efficiency=0.8, min_tire_health=0.15):
        self.base_trust = np.array(base_trust)
        self.laps = len(base_trust)
        self.regen_efficiency = regen_efficiency
        self.min_tire_health = min_tire_health
        
    def simulate_stint(self, u, b):
        """
        Simulates the tire health, battery, and lap times for given controls.
        """
        N = self.laps
        h = np.zeros(N)
        E = np.zeros(N)
        lap_times = np.zeros(N)
        
        curr_h = 1.0
        curr_E = 4.0
        
        for k in range(N):
            h[k] = curr_h
            E[k] = curr_E
            
            # Lap time model:
            # - More throttle (u) reduces time by 3.0s per unit delta
            # - ERS boost (b) reduces time by 2.0s per unit delta
            # - Tire wear (1 - h) adds up to 4.0s of degradation time loss
            lap_times[k] = (90.0 - 5.0 * self.base_trust[k] 
                            - 3.0 * (u[k] - 1.0) 
                            - 2.0 * b[k] 
                            + 4.0 * (1.0 - curr_h))
            
            # State dynamics:
            # - Tire wear degrades quadratically with push and linearly with boost
            curr_h = max(0.0, curr_h - 0.008 * (u[k] ** 2) - 0.003 * b[k])
            # - Battery is charged by regeneration and depleted by boost deployment
            curr_E = min(4.0, max(0.0, curr_E + 0.8 * self.regen_efficiency - 1.2 * b[k]))
            
        return h, E, lap_times, curr_h

    def optimize_stint(self):
        """
        Solves the trajectory optimization problem using SLSQP.
        """
        N = self.laps
        # Initial guess: moderate push (1.0) and small boost (0.1)
        x0 = np.concatenate([np.ones(N), np.full(N, 0.1)])
        
        # Bounds: u in [0.85, 1.15], b in [0.0, 1.0]
        bounds = []
        for _ in range(N):
            bounds.append((0.85, 1.15))
        for _ in range(N):
            bounds.append((0.0, 1.0))
            
        # Objective: minimize total cumulative lap time
        def objective(x):
            u = x[:N]
            b = x[N:]
            _, _, lap_times, _ = self.simulate_stint(u, b)
            return float(np.sum(lap_times))
            
        # Inequality Constraint 1: Final tire health must be above target
        def constraint_tire(x):
            u = x[:N]
            b = x[N:]
            _, _, _, final_h = self.simulate_stint(u, b)
            return float(final_h - self.min_tire_health)
            
        # Inequality Constraint 2: Battery charge must remain positive at all laps
        def constraint_battery(x):
            u = x[:N]
            b = x[N:]
            _, E, _, _ = self.simulate_stint(u, b)
            return float(np.min(E))
            
        cons = [
            {'type': 'ineq', 'fun': constraint_tire},
            {'type': 'ineq', 'fun': constraint_battery}
        ]
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        # Extract optimal profiles
        u_opt = res.x[:N]
        b_opt = res.x[N:]
        h_opt, E_opt, lap_times_opt, _ = self.simulate_stint(u_opt, b_opt)
        
        return {
            'u': u_opt,
            'b': b_opt,
            'h': h_opt,
            'E': E_opt,
            'lap_times': lap_times_opt,
            'success': bool(res.success)
        }
