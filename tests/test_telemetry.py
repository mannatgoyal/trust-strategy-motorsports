import unittest
import pandas as pd
from src.telemetry import F1TelemetryPipeline

class TestF1TelemetryPipeline(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = F1TelemetryPipeline(year=2021, track="Abu Dhabi", driver="HAM")
        
    def test_synthesized_data_fields(self):
        df = self.pipeline.generate_synthesized_data(laps=30)
        self.assertEqual(len(df), 30)
        self.assertIn('Sector1', df.columns)
        self.assertIn('Sector2', df.columns)
        self.assertIn('Sector3', df.columns)
        self.assertIn('TopSpeed', df.columns)
        self.assertIn('Throttle', df.columns)
        self.assertIn('TyreCompound', df.columns)
        self.assertTrue(all(df['LapTime'] > 0.0))

if __name__ == '__main__':
    unittest.main()
