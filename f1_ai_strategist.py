import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import custom core modules
from src.config import CONFIG
from src.telemetry import F1TelemetryPipeline
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
st.caption("SOLID Game Theory, Thermodynamic Degradation & Machine Learning System - v7.1 | July 2026")

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
    st.subheader("Configure Track Profile")
    
    track_alias_key = "silverstone" if "British" in track else "yas_marina"
    track_cfg = CONFIG.tracks.get(track_alias_key)
    
    st.markdown(f"**Loaded Profile**: `{track_alias_key.upper()}`")
    pit_loss_val = st.number_input("Pit Lane Loss (seconds)", 10.0, 30.0, float(track_cfg.pit_loss))
    deg_scale_val = st.slider("Tire Degradation Scale", 0.5, 2.0, float(track_cfg.degradation_scale))
    overtake_index_val = st.slider("Overtaking Index", 0.1, 2.5, float(track_cfg.overtaking_index))
    
    # Override CONFIG values dynamically
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
    rl_episodes = st.slider("RL training episodes", 100, 2000, 500, step=100)
    mc_trials = st.selectbox("Monte Carlo trials", [100, 500, 1000], index=1)

CONFIG.tyre.soft.base_grip = c_soft_grip
CONFIG.tyre.medium.base_grip = c_med_grip
CONFIG.tyre.hard.base_grip = c_hard_grip

# ========== Telemetry pipeline Ingestion ==========
pipeline = F1TelemetryPipeline(year, track, driver)
with st.spinner("Ingesting timing sector parameters and telemetry logs..."):
    data = pipeline.load_and_preprocess()

if not data.empty:
    st.success(f"Successfully processed {len(data)} telemetry laps for {driver} at {track}.")
    
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
    
    # Setup Solvers
    strategist = GameTheoryStrategist(data)
    sc_probs = strategist.calculate_sc_probability(track)
    
    strat_conservative = np.full(len(data), 0.90)
    strat_aggressive = np.full(len(data), 1.10)
    strat_conservative[int(len(data)*0.55)] = 0.10
    strat_aggressive[int(len(data)*0.40)] = 0.10
    
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
    and exponential wear cliff-effect thresholds. Soft, Medium, Hard compounds have unique optimal temperature windows.
    """)
    
    col_c2_1, col_c2_2 = st.columns(2)
    with col_c2_1:
        # Simulate tyre grip decay for Soft vs Hard compounds
        model_soft = TireDegradationModel(compound='soft')
        model_hard = TireDegradationModel(compound='hard')
        
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
    
    sc_model = BayesianSafetyCarModel()
    sc_combined_probs = []
    sc_vsc_probs = []
    for k in range(len(data)):
        weather_st = 'Wet' if k in [12, 13, 14] else 'Dry'
        incidents = 1 if k == 0 else 0 # Lap 1 incident odds
        posteriors = sc_model.estimate_posterior_probabilities(k+1, track, weather_st, incidents)
        sc_combined_probs.append(posteriors['Combined'])
        sc_vsc_probs.append(posteriors['VSC'])
        
    col_c4_1, col_c4_2 = st.columns([2, 1])
    with col_c4_1:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=data['LapNumber'], y=sc_combined_probs, name="Combined SC/VSC threat", line=dict(color='#e10600', width=2)))
        fig_sc.add_trace(go.Scatter(x=data['LapNumber'], y=sc_vsc_probs, name="VSC probability", line=dict(color='#eab308', width=2)))
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
        base_trust=data['StrategyConfidence'].values, 
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
    
    rl_env = F1Environment(data)
    rl_agent = QLearningAgent(lr=CONFIG.rl.learning_rate, discount=CONFIG.rl.discount_factor, epsilon=CONFIG.rl.epsilon_initial)
    
    with st.spinner("Training Reinforcement Learning Agent..."):
        _, rolling_rl = train_agent(rl_env, rl_agent, episodes=rl_episodes)
        
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
        
        # Enumerate discrete state combinations
        # Lap: 0=Early, 1=Mid, 2=Late | Wear: 0=Fresh, 1=Worn, 2=Critical | Fuel: 0=Light, 1=Heavy
        for lap_val, lap_name in [(0, "Early stint"), (2, "Late stint")]:
            for wear_val, wear_name in [(0, "Fresh"), (2, "Critical")]:
                for fuel_val, fuel_name in [(0, "Light fuel"), (1, "Heavy fuel")]:
                    # State format: (lap, wear, temp, fuel, gap, weather, drs, sc, ers)
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
    
    # === Chapter 7: Monte Carlo Risk Simulation ===
    st.header("Chapter 7: Monte Carlo Risk Assessment & Expected Finishing Positions")
    st.markdown("""
    The Monte Carlo Simulator runs randomized race trials sampling driver timing noise, weather dampness, 
    Bayesian Safety Car occurrences, and traffic bottlenecks to yield strategic success probabilities and expected positions.
    """)
    
    mc_simulator = F1MonteCarloSimulator(data)
    with st.spinner(f"Running {mc_trials} Monte Carlo simulations..."):
        nash_times, nash_positions = mc_simulator.run_simulation(strat_conservative, sc_probs, trials=mc_trials)
        stack_times, stack_positions = mc_simulator.run_simulation(strat_aggressive, sc_probs, trials=mc_trials)
        
    col_c7_1, col_c7_2 = st.columns([2, 1])
    with col_c7_1:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=nash_times, name="Conservative strategy", marker_color='#3b82f6', opacity=0.6))
        fig_mc.add_trace(go.Histogram(x=stack_times, name="Aggressive strategy", marker_color='#e10600', opacity=0.6))
        fig_mc.update_layout(**PLOTLY_LAYOUT, title="Stint Completion Duration Distributions", barmode='overlay')
        st.plotly_chart(fig_mc, use_container_width=True)
    with col_c7_2:
        st.subheader("Expected Strategic Outcomes")
        
        nash_metrics = mc_simulator.calculate_risk_metrics(nash_times, nash_positions)
        stack_metrics = mc_simulator.calculate_risk_metrics(stack_times, stack_positions)
        
        outcomes_df = pd.DataFrame({
            'Risk Metric': ['Expected Stint Duration', 'Strategic Volatility (Std Dev)', 'Expected Finishing Position', 'Win Success Rate', 'Podium Probability'],
            'Conservative Profile': [
                f"{nash_metrics['mean']:.2f}s",
                f"{nash_metrics['std_dev']:.2f}s",
                f"P{nash_metrics['expected_position']:.1f}",
                f"{nash_metrics['win_probability']:.1%}",
                f"{nash_metrics['podium_probability']:.1%}"
            ],
            'Aggressive Profile': [
                f"{stack_metrics['mean']:.2f}s",
                f"{stack_metrics['std_dev']:.2f}s",
                f"P{stack_metrics['expected_position']:.1f}",
                f"{stack_metrics['win_probability']:.1%}",
                f"{stack_metrics['podium_probability']:.1%}"
            ]
        }).set_index('Risk Metric')
        st.dataframe(outcomes_df)

    st.markdown("---")

    # === Chapter 8: Real Race Replay & Strategy Audit ===
    st.header("Chapter 8: Live Race Replay & Strategy Audit")
    st.markdown("""
    This section replays the historical stint telemetry lap-by-lap, using our thermodynamic wear models, 
    Bayesian SC estimators, and Strategy Confidence networks. It audits the actual team pit decisions 
    against the AI's recommendations.
    """)
    
    replay_engine = F1RaceReplay(data, track)
    replay_df = replay_engine.execute_replay()
    
    col_c8_1, col_c8_2 = st.columns([2, 1])
    with col_c8_1:
        # Plot Confidence timeline with Pit Actions highlighted as scatter dots
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
        
        # Overtakes, pit stop timing mismatch, alignment index
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
            
        st.info("The alignment score measures how closely the team's live tactical choices matched the thermodynamic tyre and safety car risk suggestions generated by the strategist.")

else:
    st.error("No telemetry data loaded. Ensure the session selection has valid telemetry.")
