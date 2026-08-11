import unittest
import numpy as np
import pandas as pd
from src.config import CONFIG, TRACK_ALIASES
from src.safety_car import BayesianSafetyCarModel
from src.weather import DynamicWeatherSystem
from src.tire_degradation import TireDegradationModel
from src.pit_stop import PitStopSimulator
from src.traffic import F1TrafficSimulator
from src.game_theory import GameTheoryStrategist
from src.strategy_comparison import F1StrategyComparisonEngine
from src.reinforcement_learning import F1Environment, QLearningAgent

class TestF1IntegrationPipeline(unittest.TestCase):
    def setUp(self):
        self.laps = 15
        self.data = pd.DataFrame({
            'LapNumber': np.arange(1, self.laps + 1),
            'LapTime': np.full(self.laps, 90.0),
            'Sector1': np.full(self.laps, 30.0),
            'Sector2': np.full(self.laps, 35.0),
            'Sector3': np.full(self.laps, 25.0),
            'TyreCompound': ['Medium'] * self.laps,
            'TyreAge': np.arange(1, self.laps + 1, dtype=float),
            'TrackTemp': np.full(self.laps, 35.0),
            'AmbientTemp': np.full(self.laps, 25.0),
            'GapAhead': np.full(self.laps, 5.0),
            'Position': np.full(self.laps, 3.0),
            'StrategyConfidence': np.full(self.laps, 0.85),
            'ExitGap': np.full(self.laps, 5.0)
        })

    def test_pipeline_integration_flow(self):
        # 1. Weather and SC risk profile setup
        weather_model = DynamicWeatherSystem(rain_probability=0.12)
        weather_profile = weather_model.generate_profile(self.laps, seed=42)
        weather_states = [w[1] for w in weather_profile]
        
        sc_model = BayesianSafetyCarModel()
        incidents = [1 if k == 0 else 0 for k in range(self.laps)]
        risk_profile = sc_model.generate_risk_profile(self.data, "silverstone", weather_states, incidents)
        sc_probs = risk_profile["Combined"].to_numpy()
        
        # 2. Strategy evaluation
        comp_engine = F1StrategyComparisonEngine(
            data=self.data,
            track_id="silverstone",
            track_config=CONFIG.tracks["silverstone"],
            sc_probs=sc_probs,
            confidence=self.data["StrategyConfidence"].to_numpy()
        )
        
        # 3. Game Theory resolve
        strat_con = comp_engine.generate_strategy_profiles([8], pace_level=0.90)
        strat_agg = comp_engine.generate_strategy_profiles([8], pace_level=1.10)
        
        strategist = GameTheoryStrategist(self.data)
        payoff_a, payoff_b = strategist.get_payoff_matrix(strat_con, strat_agg, strat_con, strat_agg)
        self.assertEqual(payoff_a.shape, (2, 2))
        
        # 4. Monte Carlo evaluation
        decision = comp_engine.evaluate_strategy("Test Strategy", [8], strategy_profile=strat_con, trials=20)
        self.assertGreater(decision.expected_time, 0.0)
        self.assertGreater(decision.time_std, 0.0)

    def test_track_parameter_change_affects_simulation(self):
        # High degradation scale vs low degradation scale
        model_high = TireDegradationModel(compound='medium', degradation_scale=1.5)
        model_low = TireDegradationModel(compound='medium', degradation_scale=0.5)
        
        wear_high, _, _ = model_high.step_lap(push_level=1.0, track_temp=30.0, ambient_temp=20.0)
        wear_low, _, _ = model_low.step_lap(push_level=1.0, track_temp=30.0, ambient_temp=20.0)
        
        self.assertGreater(wear_high, wear_low)

    def test_pit_loss_affects_outcomes(self):
        # Simulates with high pit loss vs low pit loss
        stop_high = PitStopSimulator(pit_lane_loss=25.0).simulate_stop("silverstone")
        stop_low = PitStopSimulator(pit_lane_loss=15.0).simulate_stop("silverstone")
        
        self.assertGreater(stop_high['total_loss'], stop_low['total_loss'])

    def test_overtaking_index_affects_traffic(self):
        traffic_high = F1TrafficSimulator(overtaking_index=2.0)
        traffic_low = F1TrafficSimulator(overtaking_index=0.5)
        
        prob_high = traffic_high.calculate_overtake_probability(
            grip_self=1.1, grip_ahead=0.8, gap=0.5, drs_zone=True, closing_speed=0.2
        )
        prob_low = traffic_low.calculate_overtake_probability(
            grip_self=1.1, grip_ahead=0.8, gap=0.5, drs_zone=True, closing_speed=0.2
        )
        self.assertGreater(prob_high, prob_low)

    def test_strategy_confidence_affects_lap_time(self):
        # Verify higher StrategyConfidence results in faster Monte Carlo expected times
        data_high = self.data.copy()
        data_high['StrategyConfidence'] = np.full(self.laps, 0.95)
        
        data_low = self.data.copy()
        data_low['StrategyConfidence'] = np.full(self.laps, 0.50)
        
        comp_high = F1StrategyComparisonEngine(data=data_high, track_id="silverstone")
        comp_low = F1StrategyComparisonEngine(data=data_low, track_id="silverstone")
        
        dec_high = comp_high.evaluate_strategy("High Conf", [8], trials=20)
        dec_low = comp_low.evaluate_strategy("Low Conf", [8], trials=20)
        
        self.assertLess(dec_high.expected_time, dec_low.expected_time)

    def test_rl_generated_strategy_evaluable_by_monte_carlo(self):
        env = F1Environment(self.data, track_config=CONFIG.tracks["silverstone"])
        agent = QLearningAgent()
        
        # Generate RL strategy profile
        rl_strategy = agent.generate_strategy(env)
        self.assertEqual(len(rl_strategy), self.laps)
        
        comp_engine = F1StrategyComparisonEngine(data=self.data, track_id="silverstone")
        decision = comp_engine.evaluate_strategy(
            "RL Strategy", [], strategy_profile=rl_strategy, trials=20
        )
        self.assertGreater(decision.expected_time, 0.0)

if __name__ == '__main__':
    unittest.main()
