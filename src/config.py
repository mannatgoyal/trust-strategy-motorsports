import os
import yaml
from dataclasses import dataclass
from typing import Dict

@dataclass
class PitConfig:
    pit_entry_loss: float
    pit_exit_loss: float
    stationary_mean: float
    stationary_std: float
    cold_tire_outlap_loss: float

@dataclass
class CompoundConfig:
    base_grip: float
    wear_rate: float
    thermal_window_min: float
    thermal_window_max: float
    cliff_wear: float

@dataclass
class TyreConfig:
    soft: CompoundConfig
    medium: CompoundConfig
    hard: CompoundConfig
    friction_heat: float
    cooling_rate: float

@dataclass
class FuelConfig:
    fuel_capacity: float
    fuel_penalty_linear: float
    fuel_penalty_quadratic: float
    aero_sensitivity: float

@dataclass
class TrafficConfig:
    dirty_air_threshold: float
    dirty_air_loss_max: float
    sigmoid_overtake_scale: float
    base_defensive_factor: float

@dataclass
class SafetyCarConfig:
    vsc_prior: float
    sc_prior: float
    red_flag_prior: float
    rain_multiplier: float

@dataclass
class RLConfig:
    learning_rate: float
    discount_factor: float
    epsilon_initial: float
    epsilon_decay: float

@dataclass
class MonteCarloConfig:
    default_trials: int
    pacing_std: float

@dataclass
class TrackOverrideConfig:
    pit_loss: float
    degradation_scale: float
    overtaking_index: float
    base_sc_probability: float

@dataclass
class RaceConfig:
    pit: PitConfig
    tyre: TyreConfig
    fuel: FuelConfig
    traffic: TrafficConfig
    safety_car: SafetyCarConfig
    rl: RLConfig
    monte_carlo: MonteCarloConfig
    tracks: Dict[str, TrackOverrideConfig]

def load_config(config_path: str = "configs/race_config.yaml") -> RaceConfig:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, config_path)
    
    with open(full_path, "r") as f:
        data = yaml.safe_load(f)
        
    pit = PitConfig(**data["pit"])
    
    tyre = TyreConfig(
        soft=CompoundConfig(**data["tyre"]["soft"]),
        medium=CompoundConfig(**data["tyre"]["medium"]),
        hard=CompoundConfig(**data["tyre"]["hard"]),
        friction_heat=data["tyre"]["friction_heat"],
        cooling_rate=data["tyre"]["cooling_rate"]
    )
    
    fuel = FuelConfig(**data["fuel"])
    traffic = TrafficConfig(**data["traffic"])
    safety_car = SafetyCarConfig(**data["safety_car"])
    rl = RLConfig(**data["rl"])
    monte_carlo = MonteCarloConfig(**data["monte_carlo"])
    
    tracks_dict = {}
    for track_name, track_data in data["tracks"].items():
        tracks_dict[track_name] = TrackOverrideConfig(**track_data)
        
    return RaceConfig(
        pit=pit,
        tyre=tyre,
        fuel=fuel,
        traffic=traffic,
        safety_car=safety_car,
        rl=rl,
        monte_carlo=monte_carlo,
        tracks=tracks_dict
    )

# Expose global config singleton
CONFIG = load_config()
