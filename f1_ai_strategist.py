import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import custom backend modules
from src.game_theory import GameTheoryStrategist
from src.trust_analysis import TrustAnalyzer
from src.differential_games import F1TrajectoryOptimizer
from src.reinforcement_learning import F1Environment, QLearningAgent, train_agent
from src.monte_carlo import F1MonteCarloSimulator

# ========== Configuration ==========
AVAILABLE_RACES = [
    (2021, "Abu Dhabi Grand Prix", ["HAM", "VER"]),
    (2020, "Abu Dhabi Grand Prix", ["HAM", "VER"]),
    (2019, "Abu Dhabi Grand Prix", ["HAM", "VER"]),
    (2021, "British Grand Prix", ["HAM", "VER"]),
    (2020, "British Grand Prix", ["HAM", "VER"])
]

TRACK_ALIASES = {
    "British Grand Prix": "Silverstone",
    "Abu Dhabi Grand Prix": "Yas Marina"
}

# ========== Data Loader ==========
@st.cache_data
def load_race_data(year, track, driver):
    try:
        event = fastf1.get_event(year, track)
        race = event.get_race()
        race.load()
        
        # Make a copy of all laps to build session timeline
        all_laps = race.laps.copy()
        all_laps['Time'] = all_laps['Time'].dt.total_seconds()
        
        # Pivot session times (index = LapNumber, columns = Driver, values = Time)
        elapsed_times = all_laps.pivot(index='LapNumber', columns='Driver', values='Time')
        
        # Filter for target driver
        laps = all_laps[all_laps['Driver'] == driver].copy()
        laps['LapTime'] = laps['LapTime'].dt.total_seconds()
        if laps.empty:
            raise ValueError("Invalid lap time data")
            
        # Telemetry Normalization: Fuel Weight Correction
        total_laps = len(laps)
        fuel_capacity = 110.0  # kg F1 max capacity
        fuel_penalty = 0.03    # seconds per kg penalty
        
        # Linearly decreasing fuel load from 110kg down to 0kg
        remaining_fuel = fuel_capacity * (1.0 - laps['LapNumber'] / total_laps)
        
        # Fuel-corrected lap time = raw time minus fuel load penalty
        laps['FuelCorrectedTime'] = laps['LapTime'] - (fuel_penalty * remaining_fuel)
        
        q1 = laps['FuelCorrectedTime'].quantile(0.25)
        if q1 == 0:
            raise ValueError("Invalid corrected lap time data")
        
        # Calculate base trust score using fuel-corrected times
        laps['Trust'] = 1 - (laps['FuelCorrectedTime']/q1 - 1).abs()
        laps['Trust'] = laps['Trust'].replace([np.inf, -np.inf], np.nan)
        laps['Trust'] = laps['Trust'].ffill().bfill().clip(0, 1)
        laps['Position'] = laps['Position'].astype(float).ffill().bfill()
        
        # Pit Window Traffic / Congestion Analysis
        pit_loss = 22.0     # seconds lost under Green Flag pit lane delta
        pit_loss_sc = 12.0  # seconds lost under Safety Car pit lane delta
        exit_gaps = []
        exit_gaps_sc = []
        traffic_densities = []
        
        for idx, row in laps.iterrows():
            lap_num = row['LapNumber']
            if lap_num in elapsed_times.index:
                # Gaps of all other drivers on this lap
                other_times = elapsed_times.loc[lap_num].drop(driver, errors='ignore').dropna()
                driver_time = elapsed_times.loc[lap_num, driver]
                
                if not pd.isna(driver_time):
                    # Green Flag Release Gap
                    t_exit = driver_time + pit_loss
                    cars_ahead = other_times[other_times < t_exit]
                    exit_gap = t_exit - cars_ahead.max() if not cars_ahead.empty else 30.0
                    density = len(cars_ahead[t_exit - cars_ahead <= 2.0])
                    
                    # Safety Car Release Gap
                    t_exit_sc = driver_time + pit_loss_sc
                    cars_ahead_sc = other_times[other_times < t_exit_sc]
                    exit_gap_sc = t_exit_sc - cars_ahead_sc.max() if not cars_ahead_sc.empty else 30.0
                else:
                    exit_gap = 30.0
                    exit_gap_sc = 30.0
                    density = 0
            else:
                exit_gap = 30.0
                exit_gap_sc = 30.0
                density = 0
                
            exit_gaps.append(float(exit_gap))
            exit_gaps_sc.append(float(exit_gap_sc))
            traffic_densities.append(int(density))
            
        laps['ExitGap'] = exit_gaps
        laps['ExitGap_SC'] = exit_gaps_sc
        laps['TrafficDensity'] = traffic_densities
        
        laps = laps[['LapNumber', 'LapTime', 'FuelCorrectedTime', 'ExitGap', 'ExitGap_SC', 'TrafficDensity', 'Position', 'Trust']].dropna()
        if laps.empty:
            raise ValueError("No valid laps after cleaning.")
        return laps.reset_index(drop=True)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

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

DISCRETE_HEATMAP_SCALE = [
    [0.0, '#080c14'],
    [0.25, '#080c14'],
    [0.25, '#1a365d'],
    [0.5, '#1a365d'],
    [0.5, '#3b82f6'],
    [0.75, '#3b82f6'],
    [0.75, '#e10600'],
    [1.0, '#e10600']
]

DISCRETE_PAYOFF_SCALE = [
    [0.0, '#e10600'],
    [0.33, '#e10600'],
    [0.33, '#111622'],
    [0.66, '#111622'],
    [0.66, '#1e40af'],
    [1.0, '#1e40af']
]

# ========== Interactive Visualization Functions ==========
def plot_strategies(data, nash, stackelberg):
    """Plot interactive comparison between actual, Nash, and Stackelberg strategies"""
    fig = go.Figure()
    
    # Actual Trust
    fig.add_trace(go.Scatter(
        x=data['LapNumber'], 
        y=data['Trust'],
        name='Actual Stint Trust',
        line=dict(color='#e10600', width=3),
        mode='lines+markers',
        hovertemplate='Lap %{x}<br>Actual Trust: %{y:.3f}<extra></extra>'
    ))
    
    # Nash Equilibrium
    fig.add_trace(go.Scatter(
        x=data['LapNumber'], 
        y=nash,
        name='Nash (Conservative)',
        line=dict(color='#3b82f6', width=2, dash='dash'),
        mode='lines',
        hovertemplate='Lap %{x}<br>Nash Trust: %{y:.3f}<extra></extra>'
    ))
    
    # Stackelberg Leadership
    fig.add_trace(go.Scatter(
        x=data['LapNumber'], 
        y=stackelberg,
        name='Stackelberg (Aggressive)',
        line=dict(color='#9ca3af', width=2, dash='dashdot'),
        mode='lines',
        hovertemplate='Lap %{x}<br>Stackelberg Trust: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Strategy Configuration and Trust Profiles", font=dict(size=16, color='#f3f4f6')),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Lap Number")
    fig.update_yaxes(title_text="Trust Coefficient (0-1)", range=[0, 1.1])
    return fig

def plot_pit_exit_gaps(data):
    """Plot interactive pit exit gaps by lap showing dirty air danger zone"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['LapNumber'],
        y=data['ExitGap'],
        name='Projected Exit Gap',
        line=dict(color='#3b82f6', width=3),
        mode='lines+markers',
        hovertemplate='Pit on Lap %{x}<br>Exit Gap: %{y:.2f}s<extra></extra>'
    ))
    
    # Add a flat red line for the Dirty Air Threshold
    fig.add_hrect(
        y0=0.0, y1=1.5,
        fillcolor="#e10600", opacity=0.1,
        line_width=0,
        annotation_text="Dirty Air Window (< 1.5s)",
        annotation_position="inside top left",
        annotation_font=dict(color="#e10600", size=10)
    )
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Pit Release Gap Projections", font=dict(size=16, color='#f3f4f6'))
    )
    fig.update_xaxes(title_text="Potential Pit Stop Lap")
    fig.update_yaxes(title_text="Exit Interval (seconds)", range=[0, max(data['ExitGap'].max() + 2, 5)])
    return fig

def plot_feature_importance_heatmap(importances):
    """Create interactive feature importance heatmap matching the telemetry window dimensions"""
    if importances is None or len(importances) != 15:
        return None
    
    importance_matrix = importances.reshape(5, 3)
    
    fig = px.imshow(
        importance_matrix,
        labels=dict(x="Telemetry Parameter", y="Window Offset", color="Weight"),
        x=['LapTime', 'Position', 'Trust'],
        y=['Lap-0 (Oldest)', 'Lap-1', 'Lap-2', 'Lap-3', 'Lap-4 (Newest)'],
        color_continuous_scale=DISCRETE_HEATMAP_SCALE,
        aspect="auto"
    )
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Random Forest Window Weighting Matrix", font=dict(size=16, color='#f3f4f6')),
        coloraxis_showscale=False
    )
    return fig

def plot_game_theory_payoff(nash_payoff, stackelberg_payoff):
    """Plot interactive 2x2 game theory payoff matrix"""
    payoff_matrix = np.array([
        [nash_payoff, nash_payoff * 0.9],
        [stackelberg_payoff * 0.9, stackelberg_payoff]
    ])
    
    fig = px.imshow(
        payoff_matrix,
        x=['Conservative', 'Aggressive'],
        y=['Conservative', 'Aggressive'],
        color_continuous_scale=DISCRETE_PAYOFF_SCALE,
        zmin=0.0,
        zmax=1.0,
        labels=dict(color="Payoff")
    )
    
    # Add text labels to each cell
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, 
                y=i, 
                text=f"{payoff_matrix[i, j]:.3f}",
                showarrow=False,
                font=dict(size=14, color="#f3f4f6", weight="bold")
            )
            
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Strategy Payoff Matrix", font=dict(size=16, color='#f3f4f6')),
        coloraxis_showscale=False
    )
    fig.update_xaxes(title_text="Competitor Strategy Profile")
    fig.update_yaxes(title_text="Driver Strategy Profile")
    return fig

def plot_sc_probability(data, sc_probs):
    """Plot safety car probability across race laps"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['LapNumber'],
        y=sc_probs,
        name='Safety Car Probability',
        line=dict(color='#e10600', width=3),
        mode='lines+markers',
        hovertemplate='Lap %{x}<br>SC Prob: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Safety Car Probability Distribution", font=dict(size=16, color='#f3f4f6'))
    )
    fig.update_xaxes(title_text="Potential Pit Stop Lap")
    fig.update_yaxes(title_text="Probability (0-1)", range=[0, max(sc_probs.max() + 0.05, 0.1)])
    return fig

def plot_control_trajectories(laps_seq, u, b):
    """Plot throttle and ERS boost optimal control trajectories"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=laps_seq, y=u,
        name="Optimal Throttle/Push (u)",
        line=dict(color='#3b82f6', width=3),
        mode='lines+markers',
        hovertemplate='Lap %{x}<br>Throttle: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=laps_seq, y=b,
        name="Optimal ERS Boost (b)",
        line=dict(color='#e10600', width=3),
        mode='lines+markers',
        hovertemplate='Lap %{x}<br>ERS Boost: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Optimal Control Trajectories (Pacing & Energy)", font=dict(size=16, color='#f3f4f6')),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Lap Number")
    fig.update_yaxes(title_text="Control Level")
    return fig

def plot_state_variables(laps_seq, h, E):
    """Plot tire health and battery state transitions"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=laps_seq, y=h,
        name="Tire Health (h)",
        line=dict(color='#10b981', width=3),
        mode='lines+markers',
        hovertemplate='Lap %{x}<br>Tire Health: %{y:.2%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=laps_seq, y=E,
        name="Battery SoC (MJ)",
        line=dict(color='#eab308', width=3),
        mode='lines+markers',
        hovertemplate='Lap %{x}<br>Battery Energy: %{y:.2f} MJ<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Vehicle State Transitions", font=dict(size=16, color='#f3f4f6')),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Lap Number")
    fig.update_yaxes(title_text="State Quantities")
    return fig

def plot_rl_convergence(rolling_rewards):
    """Plot reinforcement learning agent mean reward convergence curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rolling_rewards,
        name="Rolling Mean Reward",
        line=dict(color='#e10600', width=2),
        mode='lines',
        hovertemplate='Episode %{x}<br>Reward: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Q-Learning Agent Convergence Profile", font=dict(size=16, color='#f3f4f6'))
    )
    fig.update_xaxes(title_text="Training Episode")
    fig.update_yaxes(title_text="Mean Episode Reward")
    return fig

def plot_mc_distributions(nash_times, stack_times):
    """Plot Monte Carlo simulated stint completion distributions"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=nash_times,
        name="Nash (Conservative)",
        xbins=dict(size=1.0),
        marker_color='#3b82f6',
        opacity=0.6,
        hovertemplate='Stint time: %{x}s<br>Count: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Histogram(
        x=stack_times,
        name="Stackelberg (Aggressive)",
        xbins=dict(size=1.0),
        marker_color='#e10600',
        opacity=0.6,
        hovertemplate='Stint time: %{x}s<br>Count: %{y}<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Stint Completion Time Distributions", font=dict(size=16, color='#f3f4f6')),
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Total Stint Duration (seconds)")
    fig.update_yaxes(title_text="Simulation Trial Frequency")
    return fig

# ========== Streamlit Interface & Custom Styles ==========
st.set_page_config(page_title="F1 Strategy Engineer Toolkit", layout="wide")

# Custom styling to apply sans-serif typography, black background, and high-contrast styling
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
st.caption("Game Theory & Machine Learning System - v6.0 | July 2026")

# === Controls ===
with st.sidebar:
    st.header("Session Selector")
    year = st.selectbox("Year", sorted(list(set([r[0] for r in AVAILABLE_RACES])), reverse=True))
    available_tracks = sorted(list(set([r[1] for r in AVAILABLE_RACES if r[0] == year])))
    track = st.selectbox("Track", available_tracks)
    available_drivers = sorted(list(set([d for r in AVAILABLE_RACES if r[0] == year and r[1] == track for d in r[2]])))
    driver = st.selectbox("Driver", available_drivers)
    
    st.markdown("---")
    st.caption("Database Reference: Fast-F1 API v3.8.3")
    
    st.subheader("Analysis Parameters")
    show_demo = st.checkbox("Enable Demo Simulation", value=False, 
                           help="Display simulated telemetry matrices for review")
    advanced_view = st.checkbox("Display Advanced Statistics", value=False,
                               help="Expose differential and correlation plots")
    
    st.markdown("---")
    st.subheader("Optimal Control Tuning")
    regen_eff = st.slider("ERS Regen Efficiency", min_value=0.2, max_value=1.5, value=0.8, step=0.1,
                          help="Amount of battery energy recovered during lift-and-coast per lap.")
    min_tire_target = st.slider("Min Stint Tyre Limit", min_value=0.05, max_value=0.50, value=0.15, step=0.05,
                                help="Target structural tire health threshold remaining at stint end.")
    
    st.markdown("---")
    st.subheader("Reinforcement Learning Agent Tuning")
    rl_episodes = st.slider("Training Episodes", min_value=100, max_value=2500, value=1000, step=100,
                            help="Total number of simulated training trials for the Q-learning agent.")
    rl_epsilon = st.slider("Exploration Rate (Epsilon)", min_value=0.05, max_value=0.50, value=0.15, step=0.05,
                           help="Initial rate of random search selections decaying dynamically over training.")
    
    st.markdown("---")
    st.subheader("Monte Carlo Risk Simulator")
    mc_trials = st.slider("Monte Carlo Trials", min_value=100, max_value=5000, value=1000, step=100,
                          help="Number of randomized timeline simulation runs.")

# Load Race Data
data = load_race_data(year, track, driver)

if not data.empty and not show_demo:
    # Run Game Theory Models
    strategist = GameTheoryStrategist(data)
    nash = strategist.nash_equilibrium()
    stackelberg = strategist.stackelberg_leadership()
    
    # Calculate SC probability vector
    sc_probs = strategist.calculate_sc_probability(track)
    
    # Calculate base payoffs (Green flag)
    nash_gf, stack_gf = strategist.calculate_payoff(nash, stackelberg)
    
    # Calculate expected stochastic payoffs (including safety car risk)
    nash_payoff, stackelberg_payoff = strategist.calculate_expected_payoff(nash, stackelberg, sc_probs)
    
    # Recommended pit lap metrics
    best_strategy_name = "Stackelberg (Aggressive)" if stackelberg_payoff > nash_payoff else "Nash (Conservative)"
    best_strategy_profile = stackelberg if stackelberg_payoff > nash_payoff else nash
    best_pit_lap = int(np.argmin(best_strategy_profile))
    
    # Find exit status on selected pit lap
    projected_exit_gap = data.loc[best_pit_lap, 'ExitGap']
    exit_status = "Clean Air" if projected_exit_gap >= 1.5 else "Dirty Air"
    
    # === Section 1: Dashboard KPI Cards ===
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Base Stint Trust Score", 
            value=f"{data['Trust'].mean():.3f}",
            help="Mean driver pace consistency across the loaded stint."
        )
    with col_m2:
        st.metric(
            label="Nash Expected Payoff", 
            value=f"{nash_payoff:.3f}",
            delta=f"{nash_payoff - data['Trust'].mean():+.3f}",
            help="Expected payoff coefficient incorporating safety car probabilities."
        )
    with col_m3:
        st.metric(
            label="Stackelberg Expected Payoff", 
            value=f"{stackelberg_payoff:.3f}",
            delta=f"{stackelberg_payoff - data['Trust'].mean():+.3f}",
            help="Expected payoff coefficient incorporating safety car probabilities."
        )
    with col_m4:
        st.metric(
            label=f"Projected Pit Exit (Lap {best_pit_lap})",
            value=f"{projected_exit_gap:.1f}s",
            delta=exit_status,
            delta_color="normal" if exit_status == "Clean Air" else "inverse",
            help="Time gap to the closest car ahead immediately after rejoining the track."
        )
        
    st.markdown("---")
    
    # === Chapter 1: Telemetry Normalization ===
    st.header("Chapter 1: Telemetry Normalization")
    st.markdown("""
    In Formula 1 strategy, raw lap times are deceptive. A car starts a stint with a heavy fuel load (up to 110 kg) which burns off 
    at roughly 1.6 kg per lap, making the car naturally faster over time. This weight loss masks the real tyre degradation profile.
    
    To isolate tyre grip loss, we apply an **empty-weight normalization** by subtracting the fuel load penalty (0.03 seconds per kg) 
    from the raw timing data. This creates the **Fuel-Corrected Lap Time** and the base **Stint Trust Profile**.
    """)
    
    # Plotly comparison chart
    fig_fuel = go.Figure()
    fig_fuel.add_trace(go.Scatter(
        x=data['LapNumber'], 
        y=data['LapTime'], 
        name='Raw Lap Time', 
        line=dict(color='#e10600', width=2)
    ))
    fig_fuel.add_trace(go.Scatter(
        x=data['LapNumber'], 
        y=data['FuelCorrectedTime'], 
        name='Fuel-Corrected Lap Time', 
        line=dict(color='#3b82f6', width=2)
    ))
    fig_fuel.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Telemetry Normalization (Raw vs. Fuel-Corrected)", font=dict(size=16, color='#f3f4f6'))
    )
    fig_fuel.update_xaxes(title_text="Lap Number")
    fig_fuel.update_yaxes(title_text="Lap Time (seconds)")
    
    st.plotly_chart(fig_fuel, use_container_width=True)
    st.markdown("""
    *Observation: Note how the raw lap times (red) remain relatively flat or decrease due to fuel burnoff. 
    The fuel-corrected times (blue) expose the real physical grip loss of the compound, exhibiting a rising slope as the tyre degrades.*
    """)
    
    st.markdown("---")

    # === Chapter 2: Pit Lane Gaps & Track Congestion ===
    st.header("Chapter 2: Pit Lane Gaps and Track Congestion")
    st.markdown("""
    Pitting costs a fixed track time penalty (22.0 seconds pit lane delta). The strategy engineer must target rejoining the race 
    in **Clean Air** (free of slower cars). Rejoining in traffic (< 1.5 seconds gap) exposes the car to **Dirty Air** (aerodynamic wake), 
    which cuts downforce, reduces cornering speeds, and accelerates tyre thermal degradation.
    
    The chart below computes where the driver will rejoin relative to the track field for every possible pit lap.
    """)
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.plotly_chart(plot_pit_exit_gaps(data), use_container_width=True)
    with col_p2:
        st.subheader("Exit Window Evaluation")
        st.markdown(f"""
        Pitting into the red zone (< 1.5s) triggers a **congestion penalty** (-0.15) on the stint utility.
        
        *   **Recommended Strategy**: `{best_strategy_name}`
        *   **Optimal Release Lap**: `Lap {best_pit_lap}`
        *   **Projected Gap on Release**: `{projected_exit_gap:.2f} seconds`
        *   **Calculated Air Status**: **{exit_status}**
        """)
        if exit_status == "Clean Air":
            st.success("Recommendation verified: Release window is in clean air. No traffic penalty applied.")
        else:
            st.warning("Recommendation warning: Release window falls into traffic. Stint utility is penalized by dirty air.")
            
    st.markdown("---")
    
    # === Chapter 3: Tactical Decision Optimization ===
    st.header("Chapter 3: Tactical Decision Optimization")
    st.markdown("""
    We model the strategic options of the driver and competitor as a 2-player strategic game:
    *   **Nash (Conservative)**: Represents defensive stability. The driver optimizes tyre longevity and stint pacing, assuming the competitor will counter.
    *   **Stackelberg (Aggressive)**: Represents an aggressive first-mover stance, pushing hard early to break the DRS interval, accepting a higher tyre wear rate.
    
    The utility values are adjusted dynamically: if the chosen pit lap falls within the 1.5s dirty air threshold, a congestion penalty is deducted from the strategy's payoff.
    """)
    
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.plotly_chart(plot_strategies(data, nash, stackelberg), use_container_width=True)
    with col_g2:
        st.plotly_chart(plot_game_theory_payoff(nash_payoff, stackelberg_payoff), use_container_width=True)
        
    st.subheader("Tactical Recommendations")
    st.markdown(f"""
    *   Based on telemetry optimization, the recommended strategy is **{best_strategy_name}**.
    *   **Nash Payoff**: `{nash_payoff:.3f}` | **Stackelberg Payoff**: `{stackelberg_payoff:.3f}`.
    *   *Strategic Rule*: Pitting at `Lap {best_pit_lap}` optimizes tyre age while maximizing clean air interval times on exit.
    """)
    
    # Metrics table
    metrics = pd.DataFrame({
        'Metric': ['Average Stint Trust', 'Stint Minimum Trust', 'Trust Stability'],
        'Actual': [
            data['Trust'].mean(),
            data['Trust'].min(),
            1 - data['Trust'].std()
        ],
        'Nash (Conservative)': [
            np.mean(nash),
            np.min(nash),
            1 - np.std(nash)
        ],
        'Stackelberg (Aggressive)': [
            np.mean(stackelberg),
            np.min(stackelberg),
            1 - np.std(stackelberg)
        ]
    }).set_index('Metric')
    st.dataframe(metrics.style.format("{:.3f}").background_gradient(cmap='Blues', axis=1))

    st.markdown("---")
    
    # === Chapter 4: Stint Stability Diagnostics ===
    st.header("Chapter 4: Stint Stability Diagnostics")
    st.markdown("""
    Using a Random Forest Regressor, we train a model on a rolling 5-lap window of historical telemetry to determine what combinations 
    of consistency (Lap Time), field location (Track Position), and historical pace reliability (Trust) most strongly predict stint stability.
    
    The heatmap shows the feature importance breakdown across the 5-lap window.
    """)
    
    analyzer = TrustAnalyzer()
    X, y = analyzer.create_features(data)
    
    if len(X) > 10 and len(y) > 10:
        with st.spinner("Training Random Forest model (80/20 train-test split)..."):
            analyzer.train(X, y)
            importances = analyzer.feature_importance()
            
            if importances is not None:
                col_h1, col_h2 = st.columns([2, 1])
                with col_h1:
                    st.plotly_chart(plot_feature_importance_heatmap(importances), use_container_width=True)
                with col_h2:
                    st.subheader("Model Diagnostic Metrics")
                    st.metric(
                        label="Model R² Score (Validation Quality)", 
                        value=f"{analyzer.test_score:.4f}",
                        help="Indicates how well the validation set trust variations are explained by the telemetry model."
                    )
                    
                    importance_matrix = importances.reshape(5, 3)
                    most_important_lap = np.argmax(np.sum(importance_matrix, axis=1))
                    most_important_feature = np.argmax(np.sum(importance_matrix, axis=0))
                    feature_names = ['Lap Time', 'Track Position', 'Stint Trust']
                    
                    st.markdown(f"""
                    **Diagnostic Results:**
                    *   **Dominant Parameter**: `{feature_names[most_important_feature]}`
                    *   **Critical Window Offset**: `Lap-{most_important_lap}`
                    *   *Interpretation*: Optimizing `{feature_names[most_important_feature]}` at the `Lap-{most_important_lap}` offset of the stint provides the highest predictive stability.
                    """)
    else:
        st.warning("Insufficient telemetry rows. Min 15 laps required for AI training.")
        
    st.markdown("---")
    
    # === Chapter 5: Stochastic Strategy & Safety Car Risk ===
    st.header("Chapter 5: Stochastic Strategy and Safety Car Risk")
    st.markdown("""
    Formula 1 is a highly stochastic environment. A safety car (SC) or virtual safety car (VSC) reduces track speeds, 
    cutting the track time cost of a pit stop from **22.0 seconds (Green Flag)** down to **12.0 seconds (Safety Car)**.
    
    We model this by weighting the Green Flag and Safety Car payoffs by the lap-by-lap safety car probability distribution. 
    The probability curve peaks on Lap 1 (opening lap incident risks) and rises as tyres wear out, making drivers prone to mistakes.
    """)
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        st.plotly_chart(plot_sc_probability(data, sc_probs), use_container_width=True)
    with col_s2:
        st.subheader("Expected Optimization Comparison")
        
        # Calculate optimal pit laps under Green Flag only vs Stochastic Expected
        best_gf_profile = stackelberg if stack_gf > nash_gf else nash
        optimal_pit_gf = int(np.argmin(best_gf_profile))
        
        best_stoch_profile = stackelberg if stackelberg_payoff > nash_payoff else nash
        optimal_pit_stoch = int(np.argmin(best_stoch_profile))
        
        st.markdown(f"""
        *   **Base Safety Car Risk**: `{sc_probs[0]:.3f}` (Lap 1)
        *   **Green Flag Optimal Pit Lap**: `Lap {optimal_pit_gf}` (Payoff: `{max(nash_gf, stack_gf):.3f}`)
        *   **Stochastic Expected Optimal Pit Lap**: `Lap {optimal_pit_stoch}` (Payoff: `{max(nash_payoff, stackelberg_payoff):.3f}`)
        """)
        
        if optimal_pit_gf == optimal_pit_stoch:
            st.info("No strategy shift: The optimal pit lap remains unchanged after factoring in safety car probability.")
        else:
            st.success(f"Stochastic shift detected! Safety car risk shifts the optimal pit stop window from Lap {optimal_pit_gf} to Lap {optimal_pit_stoch}.")
            
    st.markdown("---")
    
    # === Chapter 6: Continuous Control & Differential Games ===
    st.header("Chapter 6: Continuous Control and Differential Games")
    st.markdown("""
    Race strategy is not just about discrete decisions. A driver continuously optimizes throttle control $u_k$ (tyre preservation) 
    and energy deployment $b_k$ (battery deployment) to minimize lap time while keeping tire wear above structural safety thresholds 
    and battery state-of-charge positive.
    
    We solve this trajectory pacing problem using a **dynamic optimal control solver (SLSQP)**.
    """)
    
    optimizer = F1TrajectoryOptimizer(
        base_trust=data['Trust'].values, 
        regen_efficiency=regen_eff, 
        min_tire_health=min_tire_target
    )
    opt_results = optimizer.optimize_stint()
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(plot_control_trajectories(data['LapNumber'], opt_results['u'], opt_results['b']), use_container_width=True)
    with col_c2:
        st.plotly_chart(plot_state_variables(data['LapNumber'], opt_results['h'], opt_results['E']), use_container_width=True)
        
    st.markdown("""
    *Optimization Insights*:
    *   **Throttle profile**: Shows where the driver must lift-and-coast (lower throttle) to preserve tyres or push on key sectors.
    *   **ERS Boost profile**: Deploys extra electric energy when battery levels are high, and recharges (flat lines) during high-wear phases.
    """)
    
    st.markdown("---")
    
    # === Chapter 7: Reinforcement Learning Strategy Agent ===
    st.header("Chapter 7: Reinforcement Learning Strategy Agent")
    st.markdown("""
    Instead of static pay-off matrices or numerical solvers, we can train a **Reinforcement Learning Agent** using Tabular Q-Learning. 
    The agent acts in a simulated race environment where it receives rewards for raw pace and tire preservation, and penalties 
    for tire wear, blowouts, and pitting in dirty air.
    
    Over many training episodes, the agent updates its Q-table via the temporal difference Bellman update formula to learn 
    the optimal policy.
    """)
    
    # Setup and train agent
    env = F1Environment(data)
    agent = QLearningAgent(lr=0.1, discount=0.95, epsilon=rl_epsilon)
    with st.spinner("Training Reinforcement Learning Agent..."):
        rewards, rolling_rewards = train_agent(env, agent, episodes=rl_episodes)
        
    col_rl1, col_rl2 = st.columns([2, 1])
    with col_rl1:
        st.plotly_chart(plot_rl_convergence(rolling_rewards), use_container_width=True)
    with col_rl2:
        st.subheader("Optimal Policy Matrix")
        st.markdown("""
        Below is the learned optimal pacing/pitting policy mapping for the driver's current vehicle state:
        """)
        
        # Build policy map dataframe for display
        states_list = []
        policies_list = []
        action_names = {0: "Push (Aggressive)", 1: "Save (Tyres)", 2: "Pit Stop"}
        
        # Enumerate key state combinations
        # Lap bins: 0=Early, 1=Mid, 2=Late
        # Wear bins: 0=Fresh, 1=Worn, 2=Critical
        # Traffic bins: 0=Clean, 1=Traffic
        for lap_val, lap_name in [(0, "Early stint"), (1, "Mid stint"), (2, "Late stint")]:
            for wear_val, wear_name in [(0, "Fresh"), (1, "Worn"), (2, "Critical")]:
                for traffic_val, traffic_name in [(0, "Clean Air"), (1, "Traffic")]:
                    state = (lap_val, wear_val, traffic_val)
                    q_vals = [agent.get_q_value(state, a) for a in [0, 1, 2]]
                    best_action = np.argmax(q_vals)
                    states_list.append(f"{lap_name} | {wear_name} | {traffic_name}")
                    policies_list.append(action_names[best_action])
                    
        policy_df = pd.DataFrame({
            'Vehicle State (Lap | Tyre | Traffic)': states_list,
            'Recommended RL Action': policies_list
        })
        st.dataframe(policy_df, height=350)
        
    st.markdown("---")
    
    # === Chapter 8: Monte Carlo Strategy Simulation & Risk Assessment ===
    st.header("Chapter 8: Monte Carlo Strategy Simulation and Risk Assessment")
    st.markdown("""
    While expected value payoffs identify baseline strategies, strategy engineers require variance and risk profiles. 
    Using a **Monte Carlo simulator**, we run thousands of randomized race iterations.
    
    On each lap, the simulator evaluates the probability of a safety car deployment $P(\text{SC}_k)$. If triggered, 
    the pit stops and lap durations are updated dynamically. This generates a probability distribution of total stint completion times.
    """)
    
    # Setup simulator
    simulator = F1MonteCarloSimulator(data)
    with st.spinner("Running Monte Carlo Simulations..."):
        nash_times = simulator.run_simulation(nash, sc_probs, trials=mc_trials)
        stack_times = simulator.run_simulation(stackelberg, sc_probs, trials=mc_trials)
        
    col_mc1, col_mc2 = st.columns([2, 1])
    with col_mc1:
        st.plotly_chart(plot_mc_distributions(nash_times, stack_times), use_container_width=True)
    with col_mc2:
        st.subheader("Strategic Risk Assessment")
        st.markdown("""
        Variance and worst-case Value-at-Risk ($\text{VaR}_{0.95}$) comparison:
        """)
        
        nash_metrics = simulator.calculate_risk_metrics(nash_times)
        stack_metrics = simulator.calculate_risk_metrics(stack_times)
        
        risk_df = pd.DataFrame({
            'Risk Metric': ['Expected Stint Time (Mean)', 'Strategic Volatility (Std Dev)', '95% Worst-Case VaR'],
            'Nash (Conservative)': [
                f"{nash_metrics['mean']:.2f}s",
                f"{nash_metrics['std_dev']:.2f}s",
                f"{nash_metrics['var_95']:.2f}s"
            ],
            'Stackelberg (Aggressive)': [
                f"{stack_metrics['mean']:.2f}s",
                f"{stack_metrics['std_dev']:.2f}s",
                f"{stack_metrics['var_95']:.2f}s"
            ]
        }).set_index('Risk Metric')
        st.dataframe(risk_df)
        
        st.markdown(f"""
        *Interpretation*: 
        *   **Volatility (Std Dev)** represents strategic uncertainty. A higher standard deviation indicates greater vulnerability to traffic and safety car timing.
        *   **95% VaR** shows the worst-case scenario stint time (e.g. under severe traffic or delayed pit releases).
        """)
        
    st.markdown("---")
    
    # === Section 9: Circuit Constraints ===
    st.header("Circuit Parameters")
    track_characteristics = {
        "Silverstone": {"tire_deg": "High", "overtaking": "Medium", "key_sectors": "Sectors 1 & 2"},
        "Yas Marina": {"tire_deg": "Medium", "overtaking": "Low", "key_sectors": "Sector 3"}
    }
    track_info = track_characteristics.get(TRACK_ALIASES[track], 
                                          {"tire_deg": "Medium", "overtaking": "Medium", "key_sectors": "All"})
    
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.subheader(f"Track Dynamics: {TRACK_ALIASES[track]}")
        st.markdown(f"""
        - **Tyre Degradation Rate**: {track_info['tire_deg']}
        - **Overtaking Difficulty**: {track_info['overtaking']}
        - **Critical Sector Focus**: {track_info['key_sectors']}
        """)
    with col_rec2:
        st.subheader("Driver-Specific Strategy")
        if driver == "HAM":
            st.markdown(f"""
            *   **Profile**: Consistent tyre manager, excels in late-stint pacemaking.
            *   **Focus**: Target late-window pit stop to maximize overcut potential in sector {track_info['key_sectors']}.
            """)
        elif driver == "VER":
            st.markdown(f"""
            *   **Profile**: Aggressive early pace, strong defensive positioning.
            *   **Focus**: Target early-window pit stop to break the competitor's DRS gap.
            """)
        else:
            st.markdown(f"""
            *   **Profile**: Telemetry-driven standard profile.
            *   **Focus**: Prioritize tyre temperature stabilization on out-laps.
            """)
        
    # === Section 10: Advanced View ===
    if advanced_view:
        st.header("Advanced Statistical Analysis")
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.subheader("Differential Analysis")
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(x=data['LapNumber'], y=nash - data['Trust'], name='Nash vs Actual', line=dict(color='#3b82f6')))
            fig_diff.add_trace(go.Scatter(x=data['LapNumber'], y=stackelberg - data['Trust'], name='Stackelberg vs Actual', line=dict(color='#e10600')))
            fig_diff.add_hline(y=0.0, line_dash="dash", line_color="#9ca3af")
            fig_diff.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Strategy Delta vs Actual Telemetry", font=dict(size=14, color='#f3f4f6'))
            )
            fig_diff.update_xaxes(title_text="Lap Number")
            fig_diff.update_yaxes(title_text="Trust Differential")
            st.plotly_chart(fig_diff, use_container_width=True)
            
        with col_adv2:
            st.subheader("Correlation Matrix")
            corr_df = pd.DataFrame({
                'Actual': data['Trust'],
                'Nash': nash,
                'Stackelberg': stackelberg
            }).corr()
            st.dataframe(corr_df.style.format("{:.4f}").background_gradient(cmap='coolwarm'))

else:
    # === Demo Mode fallbacks ===
    st.warning("Live data unavailable or Demo Mode checked. Displaying simulated telemetry...")
    
    # Generate mock race data (50 laps)
    demo_laps = 50
    laps_seq = np.arange(demo_laps)
    mock_trust = np.clip(0.7 + 0.2 * np.sin(laps_seq / 8) - 0.005 * laps_seq, 0, 1)
    
    # Simulated Exit Gap: let's create a sinus gap profile varying between 0.2s and 12s
    simulated_gap = 4.0 + 3.5 * np.cos(laps_seq / 3.0) + 2.0 * np.sin(laps_seq / 1.5)
    simulated_gap = np.clip(simulated_gap, 0.1, 15.0)
    
    # Traffic density: higher when gap is low
    simulated_density = np.where(simulated_gap < 2.0, 2, 0)
    
    mock_data = pd.DataFrame({
        'LapNumber': laps_seq,
        'LapTime': 90.0 - 5 * mock_trust,
        'FuelCorrectedTime': 90.0 - 5 * mock_trust - 0.03 * 110.0 * (1.0 - laps_seq / demo_laps),
        'Position': np.ones(demo_laps) * 2,
        'Trust': mock_trust,
        'ExitGap': simulated_gap,
        'ExitGap_SC': simulated_gap + 10.0,
        'TrafficDensity': simulated_density
    })
    
    strategist = GameTheoryStrategist(mock_data)
    nash = strategist.nash_equilibrium()
    stackelberg = strategist.stackelberg_leadership()
    sc_probs = strategist.calculate_sc_probability("Silverstone")
    nash_payoff, stackelberg_payoff = strategist.calculate_expected_payoff(nash, stackelberg, sc_probs)
    
    # Render interactive graphs
    st.plotly_chart(plot_strategies(mock_data, nash, stackelberg), use_container_width=True)
    st.plotly_chart(plot_game_theory_payoff(nash_payoff, stackelberg_payoff), use_container_width=True)
    
    # Demo heatmap
    demo_importances = np.array([
        0.15, 0.15, 0.9, 0.9, 0.4, 0.4,
        0.9, 0.9, 0.15, 0.15, 0.6, 0.6,
        0.15, 0.15, 0.9, 0.9, 0.6, 0.6,
        0.6, 0.6, 0.6, 0.6, 0.4, 0.4,
        0.6, 0.6, 0.15, 0.15, 0.4, 0.4
    ])[:15]
    
    st.plotly_chart(plot_feature_importance_heatmap(demo_importances), use_container_width=True)
    
    # Chapter 6 in Demo Mode
    optimizer = F1TrajectoryOptimizer(
        base_trust=mock_data['Trust'].values, 
        regen_efficiency=regen_eff, 
        min_tire_health=min_tire_target
    )
    opt_results = optimizer.optimize_stint()
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(plot_control_trajectories(mock_data['LapNumber'], opt_results['u'], opt_results['b']), use_container_width=True)
    with col_c2:
        st.plotly_chart(plot_state_variables(mock_data['LapNumber'], opt_results['h'], opt_results['E']), use_container_width=True)
        
    # Chapter 7 in Demo Mode
    env = F1Environment(mock_data)
    agent = QLearningAgent(lr=0.1, discount=0.95, epsilon=rl_epsilon)
    rewards, rolling_rewards = train_agent(env, agent, episodes=rl_episodes)
    
    col_rl1, col_rl2 = st.columns([2, 1])
    with col_rl1:
        st.plotly_chart(plot_rl_convergence(rolling_rewards), use_container_width=True)
    with col_rl2:
        states_list = []
        policies_list = []
        action_names = {0: "Push (Aggressive)", 1: "Save (Tyres)", 2: "Pit Stop"}
        
        for lap_val, lap_name in [(0, "Early stint"), (1, "Mid stint"), (2, "Late stint")]:
            for wear_val, wear_name in [(0, "Fresh"), (1, "Worn"), (2, "Critical")]:
                for traffic_val, traffic_name in [(0, "Clean Air"), (1, "Traffic")]:
                    state = (lap_val, wear_val, traffic_val)
                    q_vals = [agent.get_q_value(state, a) for a in [0, 1, 2]]
                    best_action = np.argmax(q_vals)
                    states_list.append(f"{lap_name} | {wear_name} | {traffic_name}")
                    policies_list.append(action_names[best_action])
                    
        policy_df = pd.DataFrame({
            'Vehicle State (Lap | Tyre | Traffic)': states_list,
            'Recommended RL Action': policies_list
        })
        st.dataframe(policy_df, height=350)
        
    # Chapter 8 in Demo Mode
    simulator = F1MonteCarloSimulator(mock_data)
    nash_times = simulator.run_simulation(nash, sc_probs, trials=mc_trials)
    stack_times = simulator.run_simulation(stackelberg, sc_probs, trials=mc_trials)
    
    col_mc1, col_mc2 = st.columns([2, 1])
    with col_mc1:
        st.plotly_chart(plot_mc_distributions(nash_times, stack_times), use_container_width=True)
    with col_mc2:
        st.subheader("Strategic Risk Assessment")
        nash_metrics = simulator.calculate_risk_metrics(nash_times)
        stack_metrics = simulator.calculate_risk_metrics(stack_times)
        
        risk_df = pd.DataFrame({
            'Risk Metric': ['Expected Stint Time (Mean)', 'Strategic Volatility (Std Dev)', '95% Worst-Case VaR'],
            'Nash (Conservative)': [
                f"{nash_metrics['mean']:.2f}s",
                f"{nash_metrics['std_dev']:.2f}s",
                f"{nash_metrics['var_95']:.2f}s"
            ],
            'Stackelberg (Aggressive)': [
                f"{stack_metrics['mean']:.2f}s",
                f"{stack_metrics['std_dev']:.2f}s",
                f"{stack_metrics['var_95']:.2f}s"
            ]
        }).set_index('Risk Metric')
        st.dataframe(risk_df)
