import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any

class F1Environment:
    """
    Formula 1 Reinforcement Learning environment.
    
    Expanded State Space (9 Dimensions):
      - Lap Progress bin (0: Early, 1: Mid, 2: Late stint)
      - Tyre Wear bin (0: Fresh, 1: Worn, 2: Critical wear)
      - Tyre Temp bin (0: Cold, 1: Optimal operating, 2: Overheating)
      - Fuel Level bin (0: Light weight, 1: Heavy weight)
      - Traffic Gap bin (0: Clear air, 1: Dirty air interval < 1.5s)
      - Track Dampness bin (0: Dry asphalt, 1: Damp/Wet water film)
      - DRS active status (0: Inactive, 1: DRS wing open)
      - Safety Car status (0: Clear track, 1: SC/VSC active speed limit)
      - Battery ERS SoC bin (0: Low energy harvest state, 1: High energy deployable)
      
    Actions:
      - 0: Push (Aggressive throttle, higher ERS deploy, faster lap times)
      - 1: Conserve (Tyre saving lift-and-coast, ERS harvesting)
      - 2: Pit Stop (Refit compound, reset wear & temperatures, pit transit time loss)
      - 3: Defend (Hold track position, slight timing drag penalty)
      - 4: Attack (High ERS output, attempt overtake, extra fuel burn)
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.total_laps = len(data)
        self.reset()
        
    def reset(self) -> Tuple[int, ...]:
        """Resets environment state variables"""
        self.current_lap = 0
        self.tire_wear = 0.0
        self.tire_temp = 45.0 # Starting tire temp
        self.fuel_level = 110.0 # kg starting capacity
        self.dampness = 0.0
        self.sc_active = 0
        self.battery_energy = 4.0 # MJ max ERS state of charge
        return self._get_state()
        
    def _get_state(self) -> Tuple[int, ...]:
        # 1. Lap bin
        pct = self.current_lap / max(self.total_laps, 1)
        lap_bin = 0 if pct < 0.30 else (1 if pct < 0.70 else 2)
        
        # 2. Tyre Wear bin
        wear_bin = 0 if self.tire_wear < 0.15 else (1 if self.tire_wear < 0.45 else 2)
        
        # 3. Tyre Temp bin
        temp_bin = 0 if self.tire_temp < 85.0 else (1 if self.tire_temp <= 115.0 else 2)
        
        # 4. Fuel bin (Heavy vs Light)
        fuel_bin = 1 if self.fuel_level > 55.0 else 0
        
        # 5. Gap bin (Traffic wake interval < 1.5s)
        exit_gap = 30.0
        if 'ExitGap' in self.data.columns and 0 <= self.current_lap < self.total_laps:
            exit_gap = self.data.loc[self.current_lap, 'ExitGap']
        gap_bin = 1 if exit_gap < 1.5 else 0
        
        # 6. Weather dampness bin (Dry vs Damp/Wet)
        weather_bin = 1 if self.dampness > 0.30 else 0
        
        # 7. DRS bin
        drs_bin = 1 if (exit_gap < 1.0 and self.current_lap % 5 == 0) else 0
        
        # 8. Safety Car active status
        sc_bin = self.sc_active
        
        # 9. Battery ERS SoC bin
        ers_bin = 1 if self.battery_energy > 2.0 else 0
        
        return (lap_bin, wear_bin, temp_bin, fuel_bin, gap_bin, weather_bin, drs_bin, sc_bin, ers_bin)
        
    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool]:
        """
        State transition loop.
        """
        done = False
        reward = 0.0
        
        # 10% chance of weather dampness fluctuation
        if random.random() < 0.10:
            self.dampness = min(1.0, max(0.0, self.dampness + random.choice([-0.1, 0.1])))
            
        # Dynamic Safety Car draw
        self.sc_active = 1 if random.random() < 0.015 else 0
        
        # Action transitions & timing rewards
        if action == 0:  # Push
            reward += 12.0
            self.tire_wear += 0.038
            self.tire_temp = min(140.0, self.tire_temp + 10.0 - 4.0) # friction heat - ambient cooling
            self.fuel_level -= 2.2
            self.battery_energy = max(0.0, self.battery_energy - 0.8) # boost deploy
            
        elif action == 1:  # Conserve
            reward += 6.5
            self.tire_wear += 0.010
            self.tire_temp = max(35.0, self.tire_temp - 5.0) # cool tyres
            self.fuel_level -= 1.6
            self.battery_energy = min(4.0, self.battery_energy + 0.6) # harvest energy
            
        elif action == 2:  # Pit Stop
            pit_loss = 12.0 if self.sc_active == 1 else 22.0
            reward -= pit_loss
            self.tire_wear = 0.0
            self.tire_temp = 80.0 # pre-heated blankets temp
            
            # Check traffic penalty
            state = self._get_state()
            if state[4] == 1:
                reward -= 10.0
                
        elif action == 3:  # Defend
            reward += 8.0
            self.tire_wear += 0.015
            self.tire_temp = min(140.0, self.tire_temp + 2.0)
            self.fuel_level -= 1.8
            self.battery_energy = min(4.0, self.battery_energy + 0.2)
            
        elif action == 4:  # Attack
            reward += 15.0
            self.tire_wear += 0.045
            self.tire_temp = min(140.0, self.tire_temp + 15.0 - 3.0)
            self.fuel_level -= 2.6
            self.battery_energy = max(0.0, self.battery_energy - 1.2) # aggressive deploy
            
            # Overtake bonus if rejoining near another car
            state = self._get_state()
            if state[4] == 1 and random.random() < 0.40:
                reward += 10.0
                
        # Tire wear penalties
        if self.tire_wear >= 1.0:
            reward -= 150.0  # blowout penalty
            done = True
        elif self.tire_wear > 0.45:
            reward -= 6.0
            
        # Fuel exhaustion penalty
        if self.fuel_level <= 0.0:
            reward -= 200.0
            done = True
            
        # Step increment
        self.current_lap += 1
        if self.current_lap >= self.total_laps:
            done = True
            
        next_state = self._get_state()
        return next_state, reward, done

class QLearningAgent:
    """
    Tabular Q-learning agent supporting multi-dimensional state-action policies.
    """
    def __init__(self, actions=list(range(5)), lr=0.1, discount=0.95, epsilon=0.20):
        self.actions = actions
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_q_value(self, state: Tuple[int, ...], action: int) -> float:
        return self.q_table.get((state, action), 0.0)
        
    def choose_action(self, state: Tuple[int, ...]) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_vals = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_vals)
            best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            return random.choice(best_actions)
            
    def learn(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]):
        old_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.lr * (reward + self.discount * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

def train_agent(env: F1Environment, agent: QLearningAgent, episodes: int = 1000) -> Tuple[list, list]:
    rewards_history = []
    epsilon_decay = 0.996
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
        
    rolling_rewards = []
    for i in range(len(rewards_history)):
        start = max(0, i - 50)
        rolling_rewards.append(float(np.mean(rewards_history[start:i+1])))
        
    return rewards_history, rolling_rewards
