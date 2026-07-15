import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.trust_analysis import StrategyConfidenceEstimator
from src.safety_car import BayesianSafetyCarModel
from src.tire_degradation import TireDegradationModel

class F1RaceReplay:
    """
    Live Race Replay & Strategy Audit engine.
    Ingests real historical telemetry logs, simulates the race lap-by-lap, 
    and compares the AI's tactical suggestions (pit windows, compound choices, 
    safety car reactions) against the actual team decisions.
    """
    def __init__(self, data: pd.DataFrame, track_name: str):
        self.data = data.copy()
        self.track_name = track_name
        self.estimator = StrategyConfidenceEstimator()
        self.sc_model = BayesianSafetyCarModel()
        self.tyre_model = TireDegradationModel(compound='medium')
        
    def execute_replay(self) -> pd.DataFrame:
        """
        Executes the lap-by-lap strategy audit.
        Returns:
            DataFrame containing lap number, actual pace, AI confidence, 
            SC probability, and optimal recommended pit status.
        """
        replay_log = []
        q_median = self.data['LapTime'].median() if 'LapTime' in self.data.columns else 90.0
        
        # Calculate actual pit stops in the loaded data (minimum timing indicates pit stop)
        actual_pit_laps = []
        if 'LapTime' in self.data.columns:
            # Laps where lap time is extremely long represent pit stops
            for idx, row in self.data.iterrows():
                if row['LapTime'] > q_median + 15.0:
                    actual_pit_laps.append(int(row['LapNumber']))
                    
        for idx, row in self.data.iterrows():
            lap_num = int(row['LapNumber'])
            lap_time = float(row['LapTime'])
            
            # 1. Thermodynamic Wear update
            wear, temp, grip = self.tyre_model.step_lap(push_level=1.0, track_temp=35.0, ambient_temp=25.0)
            
            # 2. Performance Confidence Component Analysis
            pace_cons = 1.0 - min(1.0, abs(lap_time - q_median) / q_median)
            wear_stability = 1.0 - (wear * 0.005)
            pred_cert = 0.95 - (idx * 0.002)
            fuel_cons = 0.98
            anomaly = 0.05 if abs(lap_time - q_median) < 2.0 else 0.40
            
            conf = self.estimator.calculate_confidence(
                pace_consistency=pace_cons,
                degradation_stability=wear_stability,
                prediction_certainty=pred_cert,
                fuel_consistency=fuel_cons,
                anomaly_score=anomaly
            )
            
            # 3. Bayesian SC Threat
            weather_st = 'Wet' if (idx in [12, 13, 14]) else 'Dry'
            recent_incidents = 1 if idx == 0 else 0
            sc_est = self.sc_model.estimate_posterior_probabilities(lap_num, self.track_name, weather_st, recent_incidents)
            sc_threat = sc_est['Combined']
            
            # 4. AI Pit Window suggestion
            # AI recommends pitting if tyre wear is critical (wear > 0.60) or safety car active and wear is high
            ai_pit_recommendation = False
            if wear > 0.60 or (sc_threat > 0.35 and wear > 0.40):
                ai_pit_recommendation = True
                self.tyre_model.reset() # Reset tire for next stint
                
            actual_pit_action = "Stay Out"
            if lap_num in actual_pit_laps:
                actual_pit_action = "Pit Stop"
                self.tyre_model.reset() # Reset tire when actual team pits
                
            ai_pit_action = "Pit Stop" if ai_pit_recommendation else "Stay Out"
            
            replay_log.append({
                'Lap': lap_num,
                'ActualPace': lap_time,
                'StrategyConfidence': conf,
                'SafetyCarThreat': sc_threat,
                'TireWear': wear,
                'ActualAction': actual_pit_action,
                'AIRecommendedAction': ai_pit_action,
                'StrategyDeviation': 1.0 if actual_pit_action != ai_pit_action else 0.0
            })
            
        return pd.DataFrame(replay_log)
