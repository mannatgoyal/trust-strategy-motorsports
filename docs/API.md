# F1 Strategy Engineer Toolkit: API Reference

This document details the classes, constructors, methods, and parameters that define the modular architecture of the F1 Strategy Toolkit.

---

## 1. Telemetry Pipeline

### `F1TelemetryPipeline` (in [src/telemetry.py](../src/telemetry.py))
Handles timing ingestion, normalization, sector segmentation, and offline synthesised fallbacks.

```python
class F1TelemetryPipeline:
    def __init__(self, year: int, track: str, driver: str):
        """
        Args:
            year: Race year (e.g. 2021).
            track: Track name string.
            driver: Target driver code (e.g. 'HAM').
        """
```

#### Methods
*   `load_and_preprocess(self) -> pd.DataFrame`: Ingests FastF1 timing records, pivoting competitor times for release gap checks and resolving sectors.
*   `generate_synthesized_data(self, laps: int = 50) -> pd.DataFrame`: Synthesizes mock telemetry variables for offline presentation.

---

## 2. Physics & Dynamic Models

### `TireDegradationModel` (in [src/tire_degradation.py](../src/tire_degradation.py))
Thermodynamic compound wear accumulator.

```python
class TireDegradationModel:
    def __init__(self, compound: str = 'Medium'):
        """
        Args:
            compound: Tyre compound ('soft', 'medium', 'hard').
        """
```

#### Methods
*   `step_lap(self, push_level: float, track_temp: float, ambient_temp: float) -> Tuple[float, float, float]`: Advances wear and thermodynamic heat levels, outputting instantaneous tyre grip.
*   `calculate_lap_penalty(self, grip: float) -> float`: Calculates lap timing penalty (seconds) due to wear and sliding.

---

### `FuelModel` (in [src/fuel_model.py](../src/fuel_model.py))
Non-linear weight timing and ERS pacing fuel burn solver.

```python
class FuelModel:
    def __init__(self, total_laps: int, initial_fuel: float = None):
        """
        Args:
            total_laps: Total laps in the stint.
            initial_fuel: Fuel mass in kg (defaults to 110.0).
        """
```

#### Methods
*   `calculate_lap_burn(self, push_level: float) -> float`: Computes fuel consumption (kg) based on pacing throttle.
*   `calculate_lap_time_effect(self, remaining_fuel: float, track_mult: float = 1.0) -> float`: Calculates non-linear weight timing penalty (seconds) using quadratic curves.

---

### `F1TrafficSimulator` (in [src/traffic.py](../src/traffic.py))
Models dirty air aerodynamic drag loss and sigmoidal overtaking probability.

```python
class F1TrafficSimulator:
    def __init__(self):
        """Loads configuration coefficients from CONFIG singleton."""
```

#### Methods
*   `calculate_dirty_air_penalty(self, exit_gap: float, drs_active: bool = False) -> float`: Returns drag loss timing penalty (seconds).
*   `calculate_overtake_probability(self, grip_self: float, grip_ahead: float, gap: float, drs_zone: bool, closing_speed: float) -> float`: Returns overtaking probability using logistic sigmoid.

---

### `PitStopSimulator` (in [src/pit_stop.py](../src/pit_stop.py))
Detailed pit transit time solver with circuit modifications.

```python
class PitStopSimulator:
    def __init__(self):
        """Loads stationary crew means and variances."""
```

#### Methods
*   `simulate_stop(self, track_name: str, random_seed: int = None) -> Dict[str, float]`: Partitions transit times and out-lap tyre warmth penalty.

---

## 3. Machine Learning & Solvers

### `StrategyConfidenceEstimator` (in [src/trust_analysis.py](../src/trust_analysis.py))
Ensembles tree model variance and timing metrics into Strategy Confidence indicators.

#### Methods
*   `calculate_confidence(self, pace_consistency: float, degradation_stability: float, prediction_certainty: float, fuel_consistency: float, anomaly_score: float) -> float`: Returns 5-component weighted confidence value.
*   `train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]`: Trains and compares RF vs. GBM, returning MAE, RMSE, and R2.

---

### `F1TrajectoryOptimizer` (in [src/differential_games.py](../src/differential_games.py))
Discretizes continuous energy and pacing controls using SLSQP optimization.

```python
class F1TrajectoryOptimizer:
    def __init__(self, base_trust: np.ndarray, regen_efficiency: float = 0.8, min_tire_health: float = 0.15):
        """
        Args:
            base_trust: Performance confidence array per lap.
            regen_efficiency: ERS harvesting factor.
            min_tire_health: Target remaining tyre safety limit (default 15%).
        """
```

#### Methods
*   `optimize_stint(self) -> Dict[str, Any]`: Computes optimal throttle pacing, battery SoC energy, and tire health trajectories.
