import numpy as np
from typing import Dict, Tuple, Any
from scipy.optimize import minimize
from src.config import CONFIG

class F1TrajectoryOptimizer:
    """
    Continuous vehicle pacing and ERS energy optimization solver.
    
    Mathematical Formulation (Differential Game & Optimal Control):
    ===============================================================
    1. State Vector (S):
       - h(t): Tire health, h in [0, 1].
       - E(t): Battery State of Charge (SoC) in MJ, E in [0, 4.0].
       - T(t): Tire surface temperature in °C, T in [35, 140].
       
    2. Control Vector (U):
       - u(t): Throttle push level, u in [0.85, 1.15].
       - b(t): ERS Boost energy deployment, b in [0, 1].
       - d(t): Braking regeneration force, d in [0, 1].
       
    3. State Transition Differential Equations:
       - Tire degradation rate:
         dh/dt = -f_h(u, b, T) = -0.008 * u^2 * (1.0 + 0.02 * max(0, T - 110)) - 0.003 * b
       - Battery SoC recovery/depletion rate:
         dE/dt = f_E(d, b) = 0.8 * eta * d - 1.2 * b
       - Tire thermodynamic temperature rate:
         dT/dt = f_T(u, grip) - cooling = 12.0 * u^2 * grip - 6.0 * (T - 25.0)
         
    4. Cost Functional (J) to Minimize (Total Stint Time):
       J = Inegral_0^T_f [ LapTime_k(u_k, b_k, d_k, h_k, E_k, T_k) ] dt
       
    5. Hamiltonian Definition (H):
       H(S, U, lambda) = LapTime(u, b, d, h, E, T) + lambda_h * dh/dt + lambda_E * dE/dt + lambda_T * dT/dt
       
    6. Numerical Optimization Strategy:
       To maintain computational efficiency under live stream constraints, the continuous Hamiltonian 
       is discretized over the stint timeline and solved using Sequential Least Squares Programming (SLSQP).
    """
    def __init__(self, base_trust: np.ndarray, regen_efficiency: float = 0.8, min_tire_health: float = 0.15):
        self.base_trust = np.array(base_trust)
        self.laps = len(base_trust)
        self.regen_efficiency = regen_efficiency
        self.min_tire_health = min_tire_health
        
    def simulate_stint(self, u: np.ndarray, b: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates vehicle states over the stint.
        """
        N = self.laps
        h = np.zeros(N)
        E = np.zeros(N)
        lap_times = np.zeros(N)
        temps = np.zeros(N)
        
        curr_h = 1.0
        curr_E = 4.0
        curr_T = 45.0 # Starting tire temp
        
        for k in range(N):
            h[k] = curr_h
            E[k] = curr_E
            temps[k] = curr_T
            
            # 1. Thermal check: Warmup vs Overheating
            if curr_T < 85.0:
                thermal_factor = 1.0 - 0.12 * ((85.0 - curr_T) / 85.0) ** 2
            elif curr_T > 115.0:
                thermal_factor = 1.0 - 0.18 * ((curr_T - 115.0) / 115.0) ** 2
            else:
                thermal_factor = 1.0
                
            # Non-linear grip
            grip = float(np.clip(1.0 * thermal_factor * curr_h, 0.1, 1.25))
            
            # 2. Timing penalty calculations
            accel_benefit = 3.2 * (u[k] - 1.0) + 1.8 * b[k]
            cornering_loss = 2.5 * (1.0 - grip)
            
            lap_times[k] = (90.0 - 5.0 * self.base_trust[k] 
                            - accel_benefit 
                            + cornering_loss)
            
            # 3. State update equations
            # Friction heating scales quadratically with push
            heat_gain = 12.0 * (u[k] ** 2) * grip
            heat_loss = 6.0 * (curr_T - 25.0) # Ambient cooling
            curr_T = float(np.clip(curr_T + heat_gain - heat_loss, 35.0, 140.0))
            
            # Wear rate degrades faster under high tyre temperature
            wear_temp_modifier = 1.0 + max(0.0, curr_T - 110.0) * 0.02
            curr_h = float(max(0.0, curr_h - 0.008 * (u[k] ** 2) * wear_temp_modifier - 0.003 * b[k]))
            
            # ERS SoC: depletes under boost, recovers under braking regeneration
            curr_E = float(np.clip(
                curr_E + 0.8 * self.regen_efficiency * d[k] - 1.2 * b[k],
                0.0,
                4.0
            ))
            
        return h, E, lap_times, temps

    def optimize_stint(self) -> Dict[str, Any]:
        """
        Solves stint pacing using SLSQP optimization.
        Variables: x = [u (laps), b (laps), d (laps)] -> total 3*N variables.
        """
        N = self.laps
        x0 = np.concatenate([np.ones(N), np.full(N, 0.1), np.full(N, 0.5)])
        
        # Variable boundaries
        bounds = []
        for _ in range(N):
            bounds.append((0.85, 1.15)) # throttle bounds
        for _ in range(N):
            bounds.append((0.0, 1.0))   # ERS boost bounds
        for _ in range(N):
            bounds.append((0.0, 1.0))   # Braking recovery bounds
            
        def objective(x):
            u = x[:N]
            b = x[N:2*N]
            d = x[2*N:]
            _, _, lap_times, _ = self.simulate_stint(u, b, d)
            return float(np.sum(lap_times))
            
        def constraint_tire(x):
            u = x[:N]
            b = x[N:2*N]
            d = x[2*N:]
            h, _, _, _ = self.simulate_stint(u, b, d)
            return float(h[-1] - self.min_tire_health)
            
        def constraint_battery(x):
            u = x[:N]
            b = x[N:2*N]
            d = x[2*N:]
            _, E, _, _ = self.simulate_stint(u, b, d)
            return float(np.min(E))
            
        cons = [
            {'type': 'ineq', 'fun': constraint_tire},
            {'type': 'ineq', 'fun': constraint_battery}
        ]
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        u_opt = res.x[:N]
        b_opt = res.x[N:2*N]
        d_opt = res.x[2*N:]
        
        h_opt, E_opt, lap_times_opt, temp_opt = self.simulate_stint(u_opt, b_opt, d_opt)
        
        return {
            'u': u_opt,
            'b': b_opt,
            'd': d_opt,
            'h': h_opt,
            'E': E_opt,
            'temps': temp_opt,
            'lap_times': lap_times_opt,
            'success': bool(res.success)
        }
