import unittest
import numpy as np
import pandas as pd
from src.game_theory import GameTheoryStrategist

class TestGameTheoryOptimization(unittest.TestCase):
    
    def setUp(self):
        self.laps = 10
        self.data = pd.DataFrame({
            'LapNumber': np.arange(1, self.laps + 1),
            'Position': np.ones(self.laps) * 3,
            'Trust': np.full(self.laps, 0.8),
            'ExitGap': np.full(self.laps, 5.0)
        })
        
    def test_mixed_nash_resolution(self):
        strategist = GameTheoryStrategist(self.data)
        
        # Mock payoff matrices
        payoff_a = np.array([[0.8, 0.4], [0.3, 0.6]])
        payoff_b = np.array([[0.7, 0.3], [0.2, 0.8]])
        
        p_a, q_b = strategist.solve_mixed_nash(payoff_a, payoff_b)
        
        # Verify probabilities lie between 0.0 and 1.0
        self.assertTrue(0.0 <= p_a <= 1.0)
        self.assertTrue(0.0 <= q_b <= 1.0)

if __name__ == '__main__':
    unittest.main()
