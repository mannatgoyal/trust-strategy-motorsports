# F1 Strategy Engineer Toolkit

**Game Theory, Machine Learning, and Telemetry Optimization for Formula 1 Strategy**

---

## 1. Overview
The **F1 Strategy Engineer Toolkit** is a decision-support system designed to model, simulate, and optimize Formula 1 race strategies. By integrating high-frequency timing data, mechanical physics normalization, and multi-agent game theory, the toolkit helps race engineers evaluate stint lengths, release windows, and competitor tactics.

---

## 2. Key Mathematical & Physical Concepts

### Chapter 1: Telemetry Normalization (Fuel Weight Correction)
Raw telemetry hides compound degradation because cars get lighter as fuel burns (~1.6kg per lap). To isolate true tyre degradation, the toolkit applies empty-weight normalization:
$$\text{Pace}_{\text{normalized}} = \text{LapTime} - \beta \cdot (\text{Remaining Fuel})$$
*   $\beta$: Fuel weight penalty ($0.03\text{s}$ per kg).
*   $\text{Remaining Fuel}$: Linearly projected fuel weight starting from $110\text{kg}$ down to $0\text{kg}$ at race completion.

### Chapter 2: Pit Lane Gaps & Track Congestion (Congestion Games)
Pitting costs a fixed track delta ($22.0\text{s}$). Rejoining within a $1.5\text{s}$ window of another car forces the driver into **dirty air** (aerodynamic wake), which reduces downforce and accelerates thermal wear. The solver applies a dynamic **congestion penalty** if:
$$\text{ExitGap}\_k < 1.5\text{s}$$
$$\text{ExitGap}\_k = T\_{\text{exit}} - \max \{ T\_{j, k} \mid T\_{j, k} < T\_{\text{exit}} \}$$

### Chapter 3: Strategy Optimization (Nash vs. Stackelberg)
The tool models the competition as a two-player game:
*   **Nash Equilibrium (Conservative)**: A stable, defensive profile where both drivers optimize simultaneously assuming the other will counter.
*   **Stackelberg Leadership (Aggressive)**: An aggressive, first-mover profile pushing early to break the DRS interval, accepting higher tire wear.

### Chapter 5: Stochastic Strategy (Safety Car Modeling)
A Safety Car (SC) or Virtual Safety Car (VSC) reduces track speeds, cutting the time cost of a pit stop from $22.0\text{s}$ to $12.0\text{s}$. We compute expected payoffs across safety car probability distributions $P(\text{SC}\_k)$:
$$U\_{\text{expected}}(S_i, S_j) = (1 - P(\text{SC}\_k)) \cdot U\_{\text{green}}(S_i, S_j) + P(\text{SC}\_k) \cdot U\_{\text{sc}}(S_i, S_j)$$

---

## 3. Repository Architecture

```
├── src/
│   ├── game_theory.py       # Game solvers (Nash, Stackelberg, stochastic SC payoffs)
│   └── trust_analysis.py    # Sliding-window features & Random Forest regressor
├── tests/
│   └── test_models.py       # Automated unit tests (10 test cases)
├── f1_ai_strategist.py      # Streamlit dashboard interface
├── requirements.txt         # Package dependencies
└── LICENSE                  # License file
```

---

## 4. Installation & Setup

### Prerequisites
*   Python 3.13.x
*   A working C++ compiler (for package dependencies if compiled from source)

### Step 1: Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```
*Note: If requirements fail to build legacy wheels from source, run `pip install fastf1 streamlit scikit-learn plotly pandas numpy` to fetch compatible binary wheels.*

### Step 2: Run the Unit Tests
Verify model calculations before launching:
```bash
python -m unittest tests/test_models.py
```

### Step 3: Start the Streamlit Dashboard
Launch the visualization client locally:
```bash
streamlit run f1_ai_strategist.py
```

---

## 5. Testing & Verification
The test suite in `tests/test_models.py` covers 10 test cases:
*   Game theory initialization and strategy boundary checks.
*   Pace fuel-weight correction boundaries.
*   Dynamic dirty air congestion penalty utility drops.
*   Safety car probability calculations and stochastic expected payoff distribution boundaries.
*   Random Forest windowed feature shape creation and regressor fitting.

---

## 6. License
This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/manna/Projects/AI-ML/trust-strategy-motorsports/LICENSE) file for details.
