import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import custom core modules
from src.config import CONFIG, TRACK_ALIASES, RaceContext
from src.telemetry import F1TelemetryPipeline, TelemetryLoadError
from src.tire_degradation import TireDegradationModel
from src.fuel_model import FuelModel
from src.traffic import F1TrafficSimulator
from src.pit_stop import PitStopSimulator
from src.safety_car import BayesianSafetyCarModel
from src.weather import DynamicWeatherSystem
from src.trust_analysis import StrategyConfidenceEstimator
from src.game_theory import GameTheoryStrategist
from src.differential_games import F1TrajectoryOptimizer
from src.reinforcement_learning import F1Environment, QLearningAgent, train_agent
from src.monte_carlo import F1MonteCarloSimulator
from src.race_replay import F1RaceReplay
from src.strategy_comparison import F1StrategyComparisonEngine
from src.strategy_engine import StrategyDecision, StrategyDecisionPolicy

# ========== Streamlit Configuration & Styling ==========
st.set_page_config(page_title="F1 Strategy Engineer Toolkit", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background-color: #080c14 !important;
    color: #f3f4f6 !important;
}
div[data-testid="stSidebar"] {
    background-color: #111622 !important;
    border-right: 1px solid #1a365d !important;
}
div[data-testid="stMetric"], div[data-testid="metric-container"] {
    background-color: #111622 !important;
    border: 1px solid #1a365d !important;
    border-left: 4px solid #e10600 !important;
    padding: 15px !important;
    border-radius: 4px !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1a365d !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #f3f4f6 !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
}
.reportview-container .main .block-container{
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("F1 Strategy Engineer Toolkit")
st.caption("SOLID Game Theory, Thermodynamic Degradation & Machine Learning System - v8.0 | August 2026")

# ========== Global Plot Styling Config (No Gradients) ==========
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#111622',
    plot_bgcolor='#080c14',
    font=dict(color='#f3f4f6', family='Inter, sans-serif'),
    xaxis=dict(
        gridcolor='#1f2937', 
        tickfont=dict(size=10, color='#9ca3af'),
        showgrid=True
    ),
    yaxis=dict(
        gridcolor='#1f2937', 
        tickfont=dict(size=10, color='#9ca3af'),
        showgrid=True
    )
)

AVAILABLE_RACES = [
    (2021, "Abu Dhabi Grand Prix", ["HAM", "VER"]),
    (2021, "British Grand Prix", ["HAM", "VER"])
]

# ========== Sidebar Parameters & Configuration Overrides ==========
with st.sidebar:
    st.header("Session Settings")
    year = st.selectbox("Year", [2021])
    track = st.selectbox("Track", ["British Grand Prix", "Abu Dhabi Grand Prix"])
    driver = st.selectbox("Driver", ["HAM", "VER"])
    
    st.markdown("---")
    st.subheader("Data Engine Source")
    data_source = st.radio("Source Mode", ["Synthetic demo", "FastF1 historical"], index=0)
    random_seed = st.number_input("Deterministic Seed", 1, 1000, 42)
    
    st.markdown("---")
    st.subheader("Configure Track Profile")
    
    track_alias_key = TRACK_ALIASES.get(track, "silverstone")
    track_cfg = CONFIG.tracks.get(track_alias_key)
    
    st.markdown(f"**Loaded Profile**: `{track_alias_key.upper()}`")
    pit_loss_val = st.number_input("Pit Lane Loss (seconds)", 10.0, 30.0, float(track_cfg.pit_loss))
    deg_scale_val = st.slider("Tire Degradation Scale", 0.5, 2.0, float(track_cfg.degradation_scale))
    overtake_index_val = st.slider("Overtaking Index", 0.1, 2.5, float(track_cfg.overtaking_index))
    
    track_cfg.pit_loss = pit_loss_val
    track_cfg.degradation_scale = deg_scale_val
    track_cfg.overtaking_index = overtake_index_val
    
    st.markdown("---")
    st.subheader("Tyre Grip Overrides")
    c_soft_grip = st.slider("Soft Grip Coefficient", 0.90, 1.20, CONFIG.tyre.soft.base_grip)
    c_med_grip = st.slider("Medium Grip Coefficient", 0.85, 1.15, CONFIG.tyre.medium.base_grip)
    c_hard_grip = st.slider("Hard Grip Coefficient", 0.80, 1.10, CONFIG.tyre.hard.base_grip)
    
    st.markdown("---")
    st.subheader("Strategic Tuning")
    risk_aversion = st.slider("Risk Aversion Coefficient (λ)", 0.0, 3.0, 1.0, step=0.1)
    rl_episodes = st.slider("RL training episodes", 100, 2000, 500, step=100)
    mc_trials = st.selectbox("Monte Carlo trials", [100, 500, 1000], index=1)

CONFIG.tyre.soft.base_grip = c_soft_grip
CONFIG.tyre.medium.base_grip = c_med_grip
CONFIG.tyre.hard.base_grip = c_hard_grip

# ========== Telemetry Ingestion / Synthesis ==========
pipeline = F1TelemetryPipeline(year, track, driver)
data = pd.DataFrame()

if data_source == "Synthetic demo":
    data = pipeline.generate_synthesized_data()
    st.info("DATA SOURCE: SYNTHETIC DEMO")
else:
    try:
        with st.spinner("Ingesting timing sector parameters and telemetry logs from FastF1..."):
            data = pipeline.load_and_preprocess()
        st.success(f"Successfully processed {len(data)} telemetry laps for {driver} at {track}.")
    except TelemetryLoadError as exc:
        st.error(f"FastF1 Ingestion failed: {exc}. Please select 'Synthetic demo' in the sidebar.")
        st.stop()

if not data.empty:
    # Calculate non-linear fuel corrections
    fuel_model = FuelModel(total_laps=len(data))
    corrected_times = []
    fuel_masses = []
    
    curr_fuel = CONFIG.fuel.fuel_capacity
    for idx, row in data.iterrows():
        fuel_loss = fuel_model.calculate_lap_time_effect(curr_fuel)
        corrected_times.append(row['LapTime'] - fuel_loss)
        fuel_masses.append(curr_fuel)
        curr_fuel = max(0.0, curr_fuel - fuel_model.calculate_lap_burn(push_level=1.0))
        
    data['FuelCorrectedTime'] = corrected_times
    data['RemainingFuel'] = fuel_masses
    
    # Evaluate Performance Confidence
    estimator = StrategyConfidenceEstimator()
    confidence_values = []
    
    q_median = data['FuelCorrectedTime'].median()
    for idx, row in data.iterrows():
        pace_cons = 1.0 - min(1.0, abs(row['FuelCorrectedTime'] - q_median) / q_median)
        wear_stability = 1.0 - (row['TyreAge'] * 0.005)
        pred_cert = 0.95 - (idx * 0.002)
        fuel_cons = 0.98
        anomaly = 0.05 if abs(row['FuelCorrectedTime'] - q_median) < 2.0 else 0.40
        
        conf = estimator.calculate_confidence(pace_cons, wear_stability, pred_cert, fuel_cons, anomaly)
        confidence_values.append(conf)
        
    data['StrategyConfidence'] = confidence_values
    
    # === Coherent Weather & Safety Car Risk Realization ===
    weather_model = DynamicWeatherSystem(rain_probability=0.12)
    weather_profile = weather_model.generate_profile(len(data), seed=random_seed)
    weather_states = [w[1] for w in weather_profile]
    
    sc_model = BayesianSafetyCarModel()
    incidents = [1 if k == 0 else 0 for k in range(len(data))] # opening lap crash likelihood
    risk_profile = sc_model.generate_risk_profile(
        data=data,
        track_name=track_alias_key,
        weather_states=weather_states,
        incidents=incidents
    )
    sc_probs = risk_profile["Combined"].to_numpy()
    
    # Create the unified RaceContext
    race_context = RaceContext(
        year=year,
        track_id=track_alias_key,
        track_config=track_cfg,
        telemetry=data,
        confidence=data['StrategyConfidence'].to_numpy(),
        sc_probability=sc_probs,
        weather_profile=weather_profile
    )
    
    # Setup Comparison Engine
    comp_engine = F1StrategyComparisonEngine(
        data=data,
        track_id=track_alias_key,
        track_config=track_cfg,
        sc_probs=sc_probs,
        confidence=race_context.confidence
    )
    
    # ========== Metric Cards Display ==========
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Mean Strategy Confidence",
            value=f"{data['StrategyConfidence'].mean():.3%}",
            help="Weighted performance consistency and model reliability score."
        )
    with col_m2:
        st.metric(
            label="Track Degradation Scale",
            value=f"{track_cfg.degradation_scale:.2f}x",
            help="Tire wear rate modifier based on track layout."
        )
    with col_m3:
        st.metric(
            label="Ambient Air Temperature",
            value=f"{data['AmbientTemp'].mean():.1f} °C",
            help="Average track-side atmospheric weather sensor reading."
        )
    with col_m4:
        st.metric(
            label="Projected Fuel Burned",
            value=f"{110.0 - data['RemainingFuel'].iloc[-1]:.1f} kg",
            help="Total fuel mass consumed over the loaded stint timeline."
        )
        
    st.markdown("---")
    
    # === Chapter 1: Telemetry Pipeline ===
    st.header("Chapter 1: Telemetry Pipeline & Normalization")
    st.markdown("""
    Formula 1 timing telemetry is processed using a multi-sector empty-weight pipeline. 
    Using configurable quadratic and linear weight timing coefficients, we normalize out the fuel weight to expose the real grip potential.
    """)
    
    col_c1_1, col_c1_2 = st.columns([2, 1])
    with col_c1_1:
        fig_telemetry = go.Figure()
        fig_telemetry.add_trace(go.Scatter(x=data['LapNumber'], y=data['LapTime'], name="Raw Timing", line=dict(color='#e10600', width=2)))
        fig_telemetry.add_trace(go.Scatter(x=data['LapNumber'], y=data['FuelCorrectedTime'], name="Fuel-Corrected", line=dict(color='#3b82f6', width=2)))
        fig_telemetry.update_layout(**PLOTLY_LAYOUT, title="Stint Lap Time Normalisation")
        st.plotly_chart(fig_telemetry, use_container_width=True)
    with col_c1_2:
        st.subheader("Timing Breakdown")
        mean_s1 = data['Sector1'].mean()
        mean_s2 = data['Sector2'].mean()
        mean_s3 = data['Sector3'].mean()
        
        st.markdown(f"""
        *   **Avg Sector 1 Split**: `{mean_s1:.3f} seconds`
        *   **Avg Sector 2 Split**: `{mean_s2:.3f} seconds`
        *   **Avg Sector 3 Split**: `{mean_s3:.3f} seconds`
        *   **Aerodynamic sensitivity coefficient**: `{CONFIG.fuel.aero_sensitivity:.5f}`
        """)
        st.info("The fuel-corrected timing profile exposes the underlying physical tyre age slope without weight interference.")

    st.markdown("---")
    
    # === Chapter 2: Thermodynamic Tyre Degradation ===
    st.header("Chapter 2: Thermodynamic Tyre Degradation")
    st.markdown("""
     Tyre grip decays thermodynamically based on friction temperature gain, track temperature cooling, 
    and exponential wear cliff-effect thresholds. Soft, Medium, and Hard compounds have unique optimal temperature windows.
    """)
    
    col_c2_1, col_c2_2 = st.columns(2)
    with col_c2_1:
        # Simulate tyre grip decay for Soft vs Hard compounds
        model_soft = TireDegradationModel(compound='soft', degradation_scale=track_cfg.degradation_scale)
        model_hard = TireDegradationModel(compound='hard', degradation_scale=track_cfg.degradation_scale)
        
        soft_grips = []
        hard_grips = []
        soft_temps = []
        hard_temps = []
        
        for k in range(len(data)):
            _, temp_s, grip_s = model_soft.step_lap(push_level=1.0, track_temp=35.0, ambient_temp=25.0)
            _, temp_h, grip_h = model_hard.step_lap(push_level=1.0, track_temp=35.0, ambient_temp=25.0)
            soft_grips.append(grip_s)
            hard_grips.append(grip_h)
            soft_temps.append(temp_s)
            hard_temps.append(temp_h)
            
        fig_grip = go.Figure()
        fig_grip.add_trace(go.Scatter(x=data['LapNumber'], y=soft_grips, name="Soft grip", line=dict(color='#e10600', width=2)))
        fig_grip.add_trace(go.Scatter(x=data['LapNumber'], y=hard_grips, name="Hard grip", line=dict(color='#9ca3af', width=2)))
        fig_grip.update_layout(**PLOTLY_LAYOUT, title="Compound Grip Coefficient Decay Profiles")
        st.plotly_chart(fig_grip, use_container_width=True)
    with col_c2_2:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=data['LapNumber'], y=soft_temps, name="Soft temp", line=dict(color='#e10600', width=2)))
        fig_temp.add_trace(go.Scatter(x=data['LapNumber'], y=hard_temps, name="Hard temp", line=dict(color='#9ca3af', width=2)))
        fig_temp.update_layout(**PLOTLY_LAYOUT, title="Compound Operating Temperatures (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("---")
    
    # === Chapter 3: Strategy Confidence Diagnostics ===
    st.header("Chapter 3: Performance/Strategy Confidence Diagnostics")
    st.markdown("""
    Instead of simple timing offsets, the **Performance Confidence** score is modeled using a 5-component weighted sum.
    We train Random Forest and Gradient Boosting models on the window features to predict stint pacing.
    """)
    
    X_ml, y_ml = estimator.create_features(data, window=5)
    metrics_ml = estimator.train_and_evaluate(X_ml, y_ml)
    
    col_c3_1, col_c3_2 = st.columns([2, 1])
    with col_c3_1:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=data['LapNumber'], y=data['StrategyConfidence'], name="Strategy Confidence", line=dict(color='#3b82f6', width=3)))
        fig_conf.update_layout(**PLOTLY_LAYOUT, title="Performance Confidence Timeline")
        st.plotly_chart(fig_conf, use_container_width=True)
    with col_c3_2:
        st.subheader("Machine Learning Performance comparison")
        if 'RF' in metrics_ml:
            ml_df = pd.DataFrame({
                'Evaluation Parameter': ['MAE (seconds)', 'RMSE (seconds)', 'R² Score'],
                'Random Forest Regressor': [
                    f"{metrics_ml['RF']['MAE']:.4f}s",
                    f"{metrics_ml['RF']['RMSE']:.4f}s",
                    f"{metrics_ml['RF']['R2']:.4%}"
                ],
                'Gradient Boosting Regressor': [
                    f"{metrics_ml['GBM']['MAE']:.4f}s",
                    f"{metrics_ml['GBM']['RMSE']:.4f}s",
                    f"{metrics_ml['GBM']['R2']:.4%}"
                ]
            }).set_index('Evaluation Parameter')
            st.dataframe(ml_df)
        else:
            st.warning("Insufficient laps for Machine Learning model validation.")

    st.markdown("---")
    
    # === Chapter 4: Bayesian Safety Car Probability ===
    st.header("Chapter 4: Bayesian Safety Car Risk Profiler")
    st.markdown("""
    F1 incidents are stochastic events. Using Bayesian prior estimation, we evaluate the dynamic lap-by-lap Safety Car, 
    VSC, and Red Flag threats based on weather states and recent crashes.
    """)
    
    col_c4_1, col_c4_2 = st.columns([2, 1])
    with col_c4_1:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=risk_profile['LapNumber'], y=risk_profile['Combined'], name="Combined SC/VSC threat", line=dict(color='#e10600', width=2)))
        fig_sc.add_trace(go.Scatter(x=risk_profile['LapNumber'], y=risk_profile['VSC'], name="VSC probability", line=dict(color='#eab308', width=2)))
        fig_sc.update_layout(**PLOTLY_LAYOUT, title="Dynamic Safety Car Bayesian Posteriors")
        st.plotly_chart(fig_sc, use_container_width=True)
    with col_c4_2:
        st.subheader("Stochastic Risk Parameters")
        st.markdown(f"""
        *   **Baseline SC Prior**: `{CONFIG.safety_car.sc_prior:.4f}`
        *   **Baseline VSC Prior**: `{CONFIG.safety_car.vsc_prior:.4f}`
        *   **Opening lap risk spike multiplier**: `3.0x`
        *   **Wet weather multiplier**: `{CONFIG.safety_car.rain_multiplier}x`
        """)
        st.info("Bayesian probabilities peak at the opening lap and spike during wet weather transitions.")

    st.markdown("---")
    
    # === Chapter 5: Continuous Control Trajectory Optimization ===
    st.header("Chapter 5: Continuous Control & Trajectory Optimization")
    st.markdown("""
    Optimal throttle pacing $u_k$ and ERS Energy Boost deployment $b_k$ are optimized using a dynamic continuous solver 
    to preserve battery charge state-of-charge and tire thermal health.
    """)
    
    optimizer = F1TrajectoryOptimizer(
        base_confidence=data['StrategyConfidence'].values, 
        regen_efficiency=0.8, 
        min_tire_health=0.15
    )
    opt_results = optimizer.optimize_stint()
    
    col_c5_1, col_c5_2 = st.columns(2)
    with col_c5_1:
        fig_opt_control = go.Figure()
        fig_opt_control.add_trace(go.Scatter(x=data['LapNumber'], y=opt_results['u'], name="Throttle (u)", line=dict(color='#3b82f6', width=2)))
        fig_opt_control.add_trace(go.Scatter(x=data['LapNumber'], y=opt_results['b'], name="ERS Boost (b)", line=dict(color='#e10600', width=2)))
        fig_opt_control.update_layout(**PLOTLY_LAYOUT, title="Optimal Control Trajectories")
        st.plotly_chart(fig_opt_control, use_container_width=True)
    with col_c5_2:
        fig_opt_states = go.Figure()
        fig_opt_states.add_trace(go.Scatter(x=data['LapNumber'], y=opt_results['h'], name="Tyre health (h)", line=dict(color='#10b981', width=2)))
        fig_opt_states.add_trace(go.Scatter(x=data['LapNumber'], y=opt_results['E'], name="Battery SoC (E)", line=dict(color='#eab308', width=2)))
        fig_opt_states.update_layout(**PLOTLY_LAYOUT, title="Optimized State Variables")
        st.plotly_chart(fig_opt_states, use_container_width=True)

    st.markdown("---")
    
    # === Chapter 6: Multi-State Reinforcement Learning ===
    st.header("Chapter 6: Multi-State Reinforcement Learning Strategy Agent")
    st.markdown("""
    An RL Agent is trained in a multi-state simulated environment (tyre compound, age, fuel level, gaps, and weather state). 
    The agent chooses between five actions: *Push, Conserve, Pit, Defend, and Attack*.
    """)
    
    rl_env = F1Environment(
        data=data,
        track_config=track_cfg,
        sc_probs=sc_probs,
        confidence=data['StrategyConfidence'].to_numpy(),
        weather_profile=weather_profile
    )
    rl_agent = QLearningAgent(lr=CONFIG.rl.learning_rate, discount=CONFIG.rl.discount_factor, epsilon=CONFIG.rl.epsilon_initial)
    
    with st.spinner("Training Reinforcement Learning Agent..."):
        _, rolling_rl = train_agent(rl_env, rl_agent, episodes=rl_episodes)
        
    rl_strategy = rl_agent.generate_strategy(rl_env)
        
    col_c6_1, col_c6_2 = st.columns([2, 1])
    with col_c6_1:
        fig_rl = go.Figure()
        fig_rl.add_trace(go.Scatter(y=rolling_rl, name="Mean rolling reward", line=dict(color='#e10600', width=2)))
        fig_rl.update_layout(**PLOTLY_LAYOUT, title="Reinforcement Learning Convergence Profiler")
        st.plotly_chart(fig_rl, use_container_width=True)
    with col_c6_2:
        st.subheader("Optimal Policy Lookup")
        states_list = []
        policies_list = []
        action_names = {0: "Push", 1: "Conserve", 2: "Pit Stop", 3: "Defend", 4: "Attack"}
        
        for lap_val, lap_name in [(0, "Early stint"), (2, "Late stint")]:
            for wear_val, wear_name in [(0, "Fresh"), (2, "Critical")]:
                for fuel_val, fuel_name in [(0, "Light fuel"), (1, "Heavy fuel")]:
                    state = (lap_val, wear_val, 1, fuel_val, 0, 0, 0, 0, 1)
                    best_action = np.argmax([rl_agent.get_q_value(state, a) for a in range(5)])
                    states_list.append(f"{lap_name} | {wear_name} | {fuel_name}")
                    policies_list.append(action_names[best_action])
                    
        policy_df = pd.DataFrame({
            'Discrete State Combo': states_list,
            'Policy Selection': policies_list
        })
        st.dataframe(policy_df, height=300)

    st.markdown("---")
    
    # === Strategic Setup and Payoffs ===
    col_comp_params1, col_comp_params2 = st.columns(2)
    with col_comp_params1:
        one_stop_target = st.slider("Strategy A (1-Stop) Pit Lap", 5, len(data)-5, int(len(data)*0.5))
    with col_comp_params2:
        two_stop_target1 = st.slider("Strategy B (2-Stop) Pit Lap 1", 5, len(data)-10, int(len(data)*0.3))
        two_stop_target2 = st.slider("Strategy B (2-Stop) Pit Lap 2", two_stop_target1+3, len(data)-5, int(len(data)*0.7))

    # Generate Candidate profiles
    strat_a_con = comp_engine.generate_strategy_profiles([one_stop_target], pace_level=0.90)
    strat_a_agg = comp_engine.generate_strategy_profiles([one_stop_target], pace_level=1.10)
    strat_b_con = comp_engine.generate_strategy_profiles([two_stop_target1, two_stop_target2], pace_level=0.90)
    strat_b_agg = comp_engine.generate_strategy_profiles([two_stop_target1, two_stop_target2], pace_level=1.10)

    # Solve Game Theory Payoffs
    strategist = GameTheoryStrategist(data)
    payoff_a, payoff_b = strategist.get_payoff_matrix(strat_a_con, strat_a_agg, strat_b_con, strat_b_agg)
    lead_idx, follow_idx = strategist.solve_stackelberg(payoff_a, payoff_b)
    nash_p_a, nash_p_b = strategist.solve_mixed_nash(payoff_a, payoff_b)

    stack_strat = strat_a_con if lead_idx == 0 else strat_a_agg
    nash_strat = strat_b_con if nash_p_b > 0.5 else strat_b_agg

    # Evaluate Strategies under uncertainty (strictly through StrategyComparisonEngine)
    with st.spinner(f"Evaluating Equilibrium and RL Strategies ({mc_trials} MC trials)..."):
        stack_decision = comp_engine.evaluate_strategy("Stackelberg Equilibrium", [one_stop_target], strategy_profile=stack_strat, trials=mc_trials)
        nash_decision = comp_engine.evaluate_strategy("Nash Equilibrium", [two_stop_target1, two_stop_target2], strategy_profile=nash_strat, trials=mc_trials)
        rl_decision = comp_engine.evaluate_strategy("RL Advisory Policy", [], strategy_profile=rl_strategy, trials=mc_trials)

    # === Chapter 7: Monte Carlo Risk Simulation ===
    st.header("Chapter 7: Monte Carlo Risk Assessment & Equilibrium Evaluations")
    st.markdown("""
    The Monte Carlo Simulator runs randomized race trials sampling driver timing noise, weather dampness, 
    Bayesian Safety Car occurrences, and traffic bottlenecks to yield strategic success probabilities and expected positions.
    """)
    
    col_c7_1, col_c7_2 = st.columns([2, 1])
    with col_c7_1:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=nash_decision.raw_times, name="Nash Equilibrium", marker_color='#3b82f6', opacity=0.6))
        fig_mc.add_trace(go.Histogram(x=stack_decision.raw_times, name="Stackelberg Equilibrium", marker_color='#e10600', opacity=0.6))
        fig_mc.add_trace(go.Histogram(x=rl_decision.raw_times, name="RL Advisory Policy", marker_color='#10b981', opacity=0.6))
        fig_mc.update_layout(**PLOTLY_LAYOUT, title="Stint Completion Duration Distributions", barmode='overlay')
        st.plotly_chart(fig_mc, use_container_width=True)
    with col_c7_2:
        st.subheader("Expected Strategic Outcomes")
        
        outcomes_df = pd.DataFrame({
            'Risk Metric': ['Expected Stint Duration', 'Strategic Volatility (Std Dev)', 'Win Success Rate', 'Podium Probability'],
            'Nash Equilibrium': [
                f"{nash_decision.expected_time:.2f}s",
                f"{nash_decision.time_std:.2f}s",
                f"{nash_decision.win_probability:.1%}",
                f"{nash_decision.podium_probability:.1%}"
            ],
            'Stackelberg Equilibrium': [
                f"{stack_decision.expected_time:.2f}s",
                f"{stack_decision.time_std:.2f}s",
                f"{stack_decision.win_probability:.1%}",
                f"{stack_decision.podium_probability:.1%}"
            ],
            'RL Advisory Policy': [
                f"{rl_decision.expected_time:.2f}s",
                f"{rl_decision.time_std:.2f}s",
                f"{rl_decision.win_probability:.1%}",
                f"{rl_decision.podium_probability:.1%}"
            ]
        }).set_index('Risk Metric')
        st.dataframe(outcomes_df)

    st.markdown("---")
    
    # === Chapter 9: Strategy Comparison Engine ===
    st.header("Chapter 9: Strategy Comparison Engine & Decision Recommendations")
    st.markdown("""
    This section evaluates Strategy A (1-Stop Baseline) against Strategy B (2-Stop Baseline) profiles, running Monte Carlo simulations 
    and generating tactical trade-off suggestions.
    """)
    
    with st.spinner("Analyzing strategy comparison..."):
        comp_results = comp_engine.compare(one_stop_target, two_stop_target1, two_stop_target2, trials=mc_trials)
        
    st.info(comp_results['recommendation_text'])
    
    col_c9_1, col_c9_2 = st.columns([2, 1])
    with col_c9_1:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Histogram(x=comp_results['times_a'], name="Strategy A (1-Stop)", marker_color='#3b82f6', opacity=0.6))
        fig_comp.add_trace(go.Scatter(x=comp_results['times_b'], name="Strategy B (2-Stop)", mode='markers', marker=dict(color='#e10600', opacity=0.4)))
        fig_comp.update_layout(**PLOTLY_LAYOUT, title="Strategy Stint Duration Comparisons", barmode='overlay')
        st.plotly_chart(fig_comp, use_container_width=True)
    with col_c9_2:
        st.subheader("Comparison Metric Table")
        comp_df = pd.DataFrame({
            'Stint Outcome': ['Expected Time (Mean)', 'Volatility (Std Dev)', 'Podium Probability'],
            'Strategy A (1-Stop)': [
                f"{comp_results['A']['mean']:.2f}s",
                f"{comp_results['A']['std_dev']:.2f}s",
                f"{comp_results['A']['podium_probability']:.1%}"
            ],
            'Strategy B (2-Stop)': [
                f"{comp_results['B']['mean']:.2f}s",
                f"{comp_results['B']['std_dev']:.2f}s",
                f"{comp_results['B']['podium_probability']:.1%}"
            ]
        }).set_index('Stint Outcome')
        st.dataframe(comp_df)

    st.markdown("---")

    # === Decision Theoretic Ranking ===
    st.header("Decision-Theoretic Recommendation Profile")
    policy = StrategyDecisionPolicy(risk_aversion=risk_aversion)
    candidates = [nash_decision, stack_decision, rl_decision, comp_results['decision_a'], comp_results['decision_b']]
    best_decision = policy.select_best(candidates)
    
    col_rec1, col_rec2 = st.columns([1, 2])
    with col_rec1:
        st.metric(
            label="RECOMMENDED STRATEGY",
            value=best_decision.strategy_name
        )
        st.metric(
            label="Decision Utility Score",
            value=f"{best_decision.score:.2f}"
        )
    with col_rec2:
        st.markdown(f"""
        ### Strategy Rationale
        *   **Selected candidate**: `{best_decision.strategy_name}`
        *   **Expected duration**: `{best_decision.expected_time:.2f}s`
        *   **Risk (Volatility σ)**: `{best_decision.time_std:.2f}s`
        *   **Win Probability**: `{best_decision.win_probability:.1%}`
        *   **Podium Probability**: `{best_decision.podium_probability:.1%}`
        *   **Upstream Confidence**: `{best_decision.confidence:.1%}`
        
        **Why**: Ranked using a risk aversion factor $\lambda$ of **{risk_aversion:.1f}** by evaluating $U = -E[T] - \lambda \sigma_T$.
        The decision engine balances faster pace against the likelihood of track-position losses due to Safety Car intervals or tire degradation.
        """)

    st.markdown("---")

    # === Chapter 10: Real Race Replay & Strategy Audit ===
    st.header("Chapter 10: Live Race Replay & Strategy Audit")
    st.markdown("""
    This section replays the historical stint telemetry lap-by-lap, using our thermodynamic wear models, 
    Bayesian SC estimators, and Strategy Confidence networks. It audits the actual team pit decisions 
    against the AI's recommendations.
    """)
    
    replay_engine = F1RaceReplay(data, track_alias_key)
    replay_df = replay_engine.execute_replay()
    
    col_c8_1, col_c8_2 = st.columns([2, 1])
    with col_c8_1:
        fig_replay = go.Figure()
        fig_replay.add_trace(go.Scatter(x=replay_df['Lap'], y=replay_df['StrategyConfidence'], name="AI Strategy Confidence", line=dict(color='#3b82f6', width=3)))
        fig_replay.add_trace(go.Scatter(x=replay_df['Lap'], y=replay_df['SafetyCarThreat'], name="SC Threat Level", line=dict(color='#eab308', width=2, dash='dash')))
        
        # Highlight Actual Team Pit Stop
        actual_pits = replay_df[replay_df['ActualAction'] == 'Pit Stop']
        fig_replay.add_trace(go.Scatter(
            x=actual_pits['Lap'], y=actual_pits['StrategyConfidence'],
            mode='markers', name="Actual Pit Stop",
            marker=dict(color='#e10600', size=12, symbol='star')
        ))
        
        # Highlight AI recommended Pit Stop
        ai_pits = replay_df[replay_df['AIRecommendedAction'] == 'Pit Stop']
        fig_replay.add_trace(go.Scatter(
            x=ai_pits['Lap'], y=ai_pits['StrategyConfidence'],
            mode='markers', name="AI Recommended Pit",
            marker=dict(color='#10b981', size=10, symbol='circle-open', line=dict(width=2))
        ))
        
        fig_replay.update_layout(**PLOTLY_LAYOUT, title="Race Replay Pacing & Pit Stop Chronology")
        st.plotly_chart(fig_replay, use_container_width=True)
    with col_c8_2:
        st.subheader("Strategy Audit Report")
        
        alignment_score = 1.0 - (replay_df['StrategyDeviation'].mean())
        pit_laps_act = list(replay_df[replay_df['ActualAction'] == 'Pit Stop']['Lap'])
        pit_laps_ai = list(replay_df[replay_df['AIRecommendedAction'] == 'Pit Stop']['Lap'])
        
        st.markdown(f"""
        *   **Actual Pit Stops executed**: `Laps {pit_laps_act}`
        *   **AI Recommended Pit Stop windows**: `Laps {pit_laps_ai}`
        *   **AI vs Team Strategy Alignment Score**: `{alignment_score:.2%}`
        """)
        
        if len(pit_laps_act) > 0 and len(pit_laps_ai) > 0:
            delay = abs(pit_laps_act[0] - pit_laps_ai[0])
            st.metric(label="Pit Decision Interval Mismatch", value=f"{delay} laps")
        else:
            st.metric(label="Pit Decision Interval Mismatch", value="0 laps")
            
        st.info("The alignment score measures how closely the team's live tactical choices matched the thermodynamic tyre and safety car risk suggestions generated by the strategy engine.")

    # === Provenance Summary Section ===
    st.markdown("---")
    st.subheader("Analytical System Provenance")
    prov_df = pd.DataFrame({
        'Provenance Field': [
            'Telemetry Data Engine', 'Race Event Target', 'Driver Profile', 
            'Stochastic Simulation Trials', 'Randomization Seed', 'Risk Model Configuration',
            'Decision Recommendation Engine'
        ],
        'Active Parameter/Source': [
            f"FastF1 historical Timing API" if data_source != "Synthetic demo" else "Offline Synthetic Generator",
            f"{year} {track}",
            f"{driver}",
            f"{mc_trials} Monte Carlo Stint runs",
            f"{random_seed}",
            f"Bayesian Safety Car Posteriors + Dynamic Weather Realization",
            f"StrategyDecisionPolicy (λ = {risk_aversion})"
        ]
    }).set_index('Provenance Field')
    st.dataframe(prov_df, use_container_width=True)

else:
    st.error("No telemetry data loaded. Ensure the session selection has valid telemetry.")
