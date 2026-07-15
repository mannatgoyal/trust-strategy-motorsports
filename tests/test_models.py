import unittest
import numpy as np
import pandas as pd
from src.config import CONFIG
from src.fuel_model import FuelModel
from src.traffic import F1TrafficSimulator
from src.pit_stop import PitStopSimulator
from src.safety_car import BayesianSafetyCarModel
from src.trust_analysis import StrategyConfidenceEstimator
from src.differential_games import F1TrajectoryOptimizer
from src.reinforcement_learning import F1Environment, QLearningAgent, train_agent
from src.monte_carlo import F1MonteCarloSimulator
from src.race_replay import F1RaceReplay
from src.strategy_comparison import F1StrategyComparisonEngine

class TestF1ModelsExpanded(unittest.TestCase):
    
    def setUp(self):
        self.laps_count = 20
        self.mock_data = pd.DataFrame({
            'LapNumber': np.arange(1, self.laps_count + 1),
            'LapTime': np.full(self.laps_count, 90.0),
            'Sector1': np.full(self.laps_count, 30.0),
            'Sector2': np.full(self.laps_count, 35.0),
            'Sector3': np.full(self.laps_count, 25.0),
            'TopSpeed': np.full(self.laps_count, 315.0),
            'AvgSpeed': np.full(self.laps_count, 225.0),
            'Throttle': np.full(self.laps_count, 80.0),
            'Brake': np.full(self.laps_count, 15.0),
            'Steering': np.zeros(self.laps_count),
            'RPM': np.full(self.laps_count, 10500.0),
            'DRS': np.zeros(self.laps_count),
            'TyreCompound': ['Medium'] * self.laps_count,
            'TyreAge': np.arange(1, self.laps_count + 1, dtype=float),
            'TrackTemp': np.full(self.laps_count, 35.0),
            'AmbientTemp': np.full(self.laps_count, 25.0),
            'RainProbability': np.zeros(self.laps_count),
            'GapAhead': np.full(self.laps_count, 5.0),
            'GapBehind': np.full(self.laps_count, 5.0),
            'Position': np.full(self.laps_count, 3.0),
            'Trust': np.full(self.laps_count, 0.85),
            'ExitGap': np.full(self.laps_count, 5.0)
        })

    def test_fuel_model_timing_penalty(self):
        model = FuelModel(total_laps=self.laps_count)
        # timing penalty at high fuel must exceed penalty at empty fuel
        penalty_full = model.calculate_lap_time_effect(110.0)
        penalty_empty = model.calculate_lap_time_effect(0.0)
        self.assertGreater(penalty_full, penalty_empty)

    def test_traffic_dirty_air_loss(self):
        simulator = F1TrafficSimulator()
        # Wake loss at 0.5s exit gap must exceed loss at 2.0s gap (clean air)
        loss_traffic = simulator.calculate_dirty_air_penalty(0.5)
        loss_clean = simulator.calculate_dirty_air_penalty(2.0)
        self.assertGreater(loss_traffic, loss_clean)

    def test_pit_stop_simulator(self):
        simulator = PitStopSimulator()
        results = simulator.simulate_stop(track_name="Silverstone")
        # Ensure total pitstop duration includes transits and stationary time
        self.assertIn('entry', results)
        self.assertIn('stationary', results)
        self.assertIn('exit', results)
        self.assertGreater(results['total_loss'], 5.0)

    def test_bayesian_safety_car_risks(self):
        model = BayesianSafetyCarModel()
        # Risk under dry weather must be lower than risk under wet weather
        risk_dry = model.estimate_posterior_probabilities(lap_num=10, track_name="Silverstone", weather_state="dry", recent_incidents=0)
        risk_wet = model.estimate_posterior_probabilities(lap_num=10, track_name="Silverstone", weather_state="wet", recent_incidents=0)
        self.assertGreater(risk_wet['Combined'], risk_dry['Combined'])

    def test_strategy_confidence_weights(self):
        estimator = StrategyConfidenceEstimator()
        # High pacing consistency yields higher performance confidence
        conf_high = estimator.calculate_confidence(0.9, 0.9, 0.9, 0.9, 0.1)
        conf_low = estimator.calculate_confidence(0.5, 0.5, 0.5, 0.5, 0.5)
        self.assertGreater(conf_high, conf_low)

    def test_trajectory_optimizer_solving(self):
        optimizer = F1TrajectoryOptimizer(
            base_trust=self.mock_data['Trust'].values, 
            regen_efficiency=0.8, 
            min_tire_health=0.15
        )
        res = optimizer.optimize_stint()
        self.assertTrue(res['success'])
        self.assertEqual(len(res['u']), self.laps_count)
        self.assertEqual(len(res['b']), self.laps_count)
        self.assertEqual(len(res['d']), self.laps_count)

    def test_rl_environment_transitions(self):
        env = F1Environment(self.mock_data)
        state = env.reset()
        # Correct discrete features layout
        self.assertEqual(len(state), 9)
        
        next_state, reward, done = env.step(action=4) # Attack action
        self.assertEqual(env.current_lap, 1)
        self.assertGreater(env.tire_wear, 0.0)

    def test_monte_carlo_risk_calculations(self):
        simulator = F1MonteCarloSimulator(self.mock_data)
        sc_probs = np.full(self.laps_count, 0.05)
        nash_strat = np.ones(self.laps_count)
        
        times, positions = simulator.run_simulation(nash_strat, sc_probs, trials=50)
        metrics = simulator.calculate_risk_metrics(times, positions)
        
        self.assertIn('mean', metrics)
        self.assertIn('std_dev', metrics)
        self.assertIn('ci_95', metrics)
        self.assertGreater(metrics['std_dev'], 0.0)

    def test_race_replay_timeline(self):
        replay = F1RaceReplay(self.mock_data, track_name="Silverstone")
        results = replay.execute_replay()
        
        self.assertEqual(len(results), self.laps_count)
        self.assertIn('StrategyConfidence', results.columns)
        self.assertIn('SafetyCarThreat', results.columns)
        self.assertIn('AIRecommendedAction', results.columns)

    def test_strategy_comparison_recommendations(self):
        sc_probs = np.full(self.laps_count, 0.05)
        engine = F1StrategyComparisonEngine(self.mock_data, sc_probs)
        results = engine.compare(one_stop_lap=10, two_stop_lap1=6, two_stop_lap2=14, trials=30)
        
        self.assertIn('A', results)
        self.assertIn('B', results)
        self.assertIn('recommendation_text', results)
        self.assertIn('recommendation_code', results)

if __name__ == '__main__':
    unittest.main()
