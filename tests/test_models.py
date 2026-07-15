import unittest
import numpy as np
import pandas as pd
from src.game_theory import GameTheoryStrategist
from src.trust_analysis import TrustAnalyzer
from src.differential_games import F1TrajectoryOptimizer
from src.reinforcement_learning import F1Environment, QLearningAgent, train_agent
from src.monte_carlo import F1MonteCarloSimulator

class TestMotorsportsModels(unittest.TestCase):
    
    def setUp(self):
        # Create a mock telemetry dataset of 25 laps
        self.laps_count = 25
        self.mock_data = pd.DataFrame({
            'LapNumber': np.arange(1, self.laps_count + 1),
            'LapTime': 90.0 - 5.0 * np.sin(np.arange(self.laps_count) / 5.0),
            'Position': np.ones(self.laps_count) * 3,
            'Trust': np.clip(0.8 + 0.1 * np.cos(np.arange(self.laps_count) / 3.0), 0.0, 1.0),
            # Default clean air gaps (e.g. 5 seconds)
            'ExitGap': np.ones(self.laps_count) * 5.0,
            'ExitGap_SC': np.ones(self.laps_count) * 15.0, # Clean exit under Safety Car
            'TrafficDensity': np.zeros(self.laps_count, dtype=int)
        })

    def test_game_theory_initialization(self):
        strategist = GameTheoryStrategist(self.mock_data)
        self.assertEqual(strategist.laps, self.laps_count)
        self.assertEqual(len(strategist.base_trust), self.laps_count)

    def test_nash_equilibrium(self):
        strategist = GameTheoryStrategist(self.mock_data)
        nash_strat = strategist.nash_equilibrium()
        
        # Verify length and boundaries
        self.assertEqual(len(nash_strat), self.laps_count)
        self.assertTrue(np.all(nash_strat >= 0.0))
        self.assertTrue(np.all(nash_strat <= 1.0))

    def test_stackelberg_leadership(self):
        strategist = GameTheoryStrategist(self.mock_data)
        stack_strat = strategist.stackelberg_leadership()
        
        # Verify length and boundaries
        self.assertEqual(len(stack_strat), self.laps_count)
        self.assertTrue(np.all(stack_strat >= 0.0))
        self.assertTrue(np.all(stack_strat <= 1.0))

    def test_calculate_payoff(self):
        strategist = GameTheoryStrategist(self.mock_data)
        nash = strategist.nash_equilibrium()
        stackelberg = strategist.stackelberg_leadership()
        
        payoff_a, payoff_b = strategist.calculate_payoff(nash, stackelberg)
        
        # Verify payoffs are floats between 0 and 1
        self.assertIsInstance(payoff_a, float)
        self.assertIsInstance(payoff_b, float)
        self.assertTrue(0.0 <= payoff_a <= 1.0)
        self.assertTrue(0.0 <= payoff_b <= 1.0)

    def test_trust_analyzer_features(self):
        analyzer = TrustAnalyzer()
        X, y = analyzer.create_features(self.mock_data, window=5)
        
        # With 25 laps and window=5, should have 20 samples
        self.assertEqual(len(X), 20)
        self.assertEqual(len(y), 20)
        # 5 laps * 3 features = 15 features per sample
        self.assertEqual(X.shape[1], 15)

    def test_trust_analyzer_training(self):
        analyzer = TrustAnalyzer()
        X, y = analyzer.create_features(self.mock_data, window=5)
        analyzer.train(X, y)
        
        # Model must be flagged as trained
        self.assertTrue(analyzer.is_trained)
        
        # Check feature importance shape (should be 15)
        importances = analyzer.feature_importance()
        self.assertIsNotNone(importances)
        self.assertEqual(len(importances), 15)
        
        # Test score should be a float representing R-squared validation metric
        self.assertIsInstance(analyzer.test_score, float)

    def test_fuel_weight_correction(self):
        # We manually apply the fuel correction logic to check boundary behavior
        total_laps = self.laps_count
        fuel_capacity = 110.0  # kg
        fuel_penalty = 0.03    # seconds per kg
        
        # Remaining fuel capacity at start (Lap 1) vs end (Lap 25)
        remaining_fuel_lap1 = fuel_capacity * (1.0 - 1 / total_laps)
        remaining_fuel_lap25 = fuel_capacity * (1.0 - 25 / total_laps)
        
        raw_lap1_time = self.mock_data.loc[0, 'LapTime']
        raw_lap25_time = self.mock_data.loc[24, 'LapTime']
        
        corrected_lap1_time = raw_lap1_time - (fuel_penalty * remaining_fuel_lap1)
        corrected_lap25_time = raw_lap25_time - (fuel_penalty * remaining_fuel_lap25)
        
        # Corrected time at start must be strictly faster (less) than raw lap time
        self.assertLess(corrected_lap1_time, raw_lap1_time)
        # Corrected time at final lap equals raw lap time (remaining fuel is 0kg)
        self.assertEqual(corrected_lap25_time, raw_lap25_time)

    def test_traffic_congestion_penalty(self):
        # 1. Define clean air data (payoff baseline)
        clean_data = self.mock_data.copy()
        strategist_clean = GameTheoryStrategist(clean_data, traffic_penalty=0.15)
        
        nash_strat = strategist_clean.nash_equilibrium()
        stack_strat = strategist_clean.stackelberg_leadership()
        
        payoff_clean = strategist_clean.calculate_utility(nash_strat, stack_strat)
        
        # 2. Define dirty air data (pitting at lap 10 has exit gap < 1.5s)
        # For Nash, the pit lap argmin is Lap 10 (int(25 * 0.4) = 10)
        dirty_data = self.mock_data.copy()
        dirty_data.loc[10, 'ExitGap'] = 0.5  # Heavy traffic!
        
        strategist_dirty = GameTheoryStrategist(dirty_data, traffic_penalty=0.15)
        payoff_dirty = strategist_dirty.calculate_utility(nash_strat, stack_strat)
        
        # Payload with traffic congestion penalty must be exactly 0.15 less than clean air baseline
        self.assertLess(payoff_dirty, payoff_clean)
        self.assertAlmostEqual(payoff_clean - payoff_dirty, 0.15, places=5)

    def test_safety_car_probability(self):
        strategist = GameTheoryStrategist(self.mock_data)
        
        # Test Silverstone (high risk) vs default tracks
        probs_silverstone = strategist.calculate_sc_probability("Silverstone")
        probs_default = strategist.calculate_sc_probability("Yas Marina")
        
        self.assertEqual(len(probs_silverstone), self.laps_count)
        self.assertEqual(len(probs_default), self.laps_count)
        
        # Silverstone baseline (index 1) must be higher than Abu Dhabi base
        self.assertGreater(probs_silverstone[1], probs_default[1])
        
        # Lap 1 (index 0) must exhibit a risk spike
        self.assertGreater(probs_silverstone[0], probs_silverstone[1])

    def test_expected_stochastic_payoffs(self):
        # Setup data where standard pit stop hits traffic but SC pit stop gets clean air
        test_data = self.mock_data.copy()
        test_data.loc[10, 'ExitGap'] = 0.5     # Dirty air under standard
        test_data.loc[10, 'ExitGap_SC'] = 3.0  # Clean air under SC
        
        strategist = GameTheoryStrategist(test_data, traffic_penalty=0.20)
        nash = strategist.nash_equilibrium()
        stack = strategist.stackelberg_leadership()
        
        # Create a mock safety car probability vector
        sc_probs = np.zeros(self.laps_count)
        sc_probs[10] = 0.40  # 40% chance of Safety Car on pit lap 10
        
        expected_a, expected_b = strategist.calculate_expected_payoff(nash, stack, sc_probs)
        
        ut_green = strategist.calculate_utility(nash, stack, sc_active=False)
        ut_sc = strategist.calculate_utility(nash, stack, sc_active=True)
        
        # Expected utility must be exactly the weighted average: 0.6 * green + 0.4 * sc
        weighted_average = 0.6 * ut_green + 0.4 * ut_sc
        self.assertAlmostEqual(expected_a, weighted_average, places=5)

    def test_trajectory_optimizer_solving(self):
        optimizer = F1TrajectoryOptimizer(
            base_trust=self.mock_data['Trust'].values, 
            regen_efficiency=0.8, 
            min_tire_health=0.15
        )
        res = optimizer.optimize_stint()
        
        # Verify result contains the optimal profiles
        self.assertTrue(res['success'])
        self.assertEqual(len(res['u']), self.laps_count)
        self.assertEqual(len(res['b']), self.laps_count)
        self.assertEqual(len(res['h']), self.laps_count)
        self.assertEqual(len(res['E']), self.laps_count)

    def test_trajectory_optimizer_constraints(self):
        min_tire = 0.20
        optimizer = F1TrajectoryOptimizer(
            base_trust=self.mock_data['Trust'].values, 
            regen_efficiency=0.6, 
            min_tire_health=min_tire
        )
        res = optimizer.optimize_stint()
        
        # 1. Final tire health constraint verification (must be >= min_tire - tolerance)
        final_h = res['h'][-1]
        self.assertGreaterEqual(final_h, min_tire - 1e-4)
        
        # 2. Battery SoC constraint verification (must remain non-negative and <= 4.0)
        self.assertTrue(np.all(res['E'] >= -1e-4))
        self.assertTrue(np.all(res['E'] <= 4.0 + 1e-4))

    def test_rl_environment_transitions(self):
        env = F1Environment(self.mock_data)
        state = env.reset()
        
        # Initial state should be early lap, fresh tyre, clean air
        self.assertEqual(state, (0, 0, 0))
        
        # Take a Push step
        next_state, reward, done = env.step(action=0)
        self.assertEqual(env.current_lap, 1)
        self.assertAlmostEqual(env.tire_wear, 0.035, places=5)
        
        # Incur tire blowout
        env.tire_wear = 1.0
        _, reward_blowout, done_blowout = env.step(action=0)
        self.assertTrue(done_blowout)
        self.assertLess(reward_blowout, -50.0)

    def test_rl_agent_training_success(self):
        env = F1Environment(self.mock_data)
        agent = QLearningAgent(actions=[0, 1, 2], lr=0.1, discount=0.9, epsilon=0.2)
        
        # Train agent over a few episodes
        episodes_count = 50
        history, rolling = train_agent(env, agent, episodes=episodes_count)
        
        self.assertEqual(len(history), episodes_count)
        self.assertEqual(len(rolling), episodes_count)
        
        # Q-table should have learned mappings
        self.assertGreater(len(agent.q_table), 0)

    def test_monte_carlo_simulation_runs(self):
        simulator = F1MonteCarloSimulator(self.mock_data)
        sc_probs = np.full(self.laps_count, 0.05)
        nash_strat = np.ones(self.laps_count)
        
        trials = 120
        results = simulator.run_simulation(nash_strat, sc_probs, trials=trials)
        
        self.assertEqual(len(results), trials)
        self.assertTrue(np.all(results > 0.0))

    def test_monte_carlo_risk_calculations(self):
        simulator = F1MonteCarloSimulator(self.mock_data)
        sc_probs = np.full(self.laps_count, 0.10)
        nash_strat = np.ones(self.laps_count)
        
        results = simulator.run_simulation(nash_strat, sc_probs, trials=100)
        metrics = simulator.calculate_risk_metrics(results)
        
        # Verify keys and positive standard deviation
        self.assertIn('mean', metrics)
        self.assertIn('std_dev', metrics)
        self.assertIn('var_95', metrics)
        self.assertGreater(metrics['std_dev'], 0.0)
        self.assertGreater(metrics['var_95'], metrics['mean'])

if __name__ == '__main__':
    unittest.main()
