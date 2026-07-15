import unittest
from src.tire_degradation import TireDegradationModel

class TestTireDegradationModel(unittest.TestCase):
    
    def test_soft_compound_degrades_faster_than_hard(self):
        soft_model = TireDegradationModel(compound='soft')
        hard_model = TireDegradationModel(compound='hard')
        
        # Step both models with identical driver push (1.1)
        soft_wear, _, _ = soft_model.step_lap(push_level=1.1, track_temp=30.0, ambient_temp=20.0)
        hard_wear, _, _ = hard_model.step_lap(push_level=1.1, track_temp=30.0, ambient_temp=20.0)
        
        # Soft wear increment must be strictly greater than Hard wear increment
        self.assertGreater(soft_wear, hard_wear)
        
    def test_cliff_effect_reduces_grip(self):
        model = TireDegradationModel(compound='medium')
        # Manually trigger the wear cliff (0.80)
        model.wear = 0.85
        _, _, grip = model.step_lap(push_level=1.0, track_temp=30.0, ambient_temp=20.0)
        
        # Grip must drop due to exponential cliff dropoff
        self.assertLess(grip, 0.70)

if __name__ == '__main__':
    unittest.main()
