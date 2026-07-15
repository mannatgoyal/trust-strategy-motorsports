import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any

class F1Environment:
    """
    Formula 1 Reinforcement Learning environment.
    States: (Lap bin, Wear bin, Fuel bin, Gap bin, Weather dampness bin, SC flag)
    Actions: 0=Push, 1=Conserve, 2=Pit, 3=Defend, 4=Attack
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.total_laps = len(data)
        self.reset()
        
    def reset(self) -> Tuple[int, ...]:
        """Resets environment state variables"""
        self.current_lap = 0
        self.tire_wear = 0.0
        self.fuel_level = 110.0 # kg starting capacity
        self.dampness = 0.0
        self.sc_active = 0
        return self._get_state()
        
    def _get_state(self) -> Tuple[int, ...]:
        # 1. Lap bin
        pct = self.current_lap / max(self.total_laps, 1)
        lap_bin = 0 if pct < 0.30 else (1 if pct < 0.70 else 2)
        
        # 2. Tyre Wear bin
        wear_bin = 0 if self.tire_wear < 0.15 else (1 if self.tire_wear < 0.45 else 2)
        
        # 3. Fuel bin (Heavy vs Light)
        fuel_bin = 1 if self.fuel_level > 55.0 else 0
        
        # 4. Gap bin (Traffic wake interval < 1.5s)
        exit_gap = 30.0
        if 'ExitGap' in self.data.columns and 0 <= self.current_lap < self.total_laps:
            exit_gap = self.data.loc[self.current_lap, 'ExitGap']
        gap_bin = 1 if exit_gap < 1.5 else 0
        
        # 5. Weather dampness bin (Dry vs Damp/Wet)
        weather_bin = 1 if self.dampness > 0.30 else 0
        
        # 6. Safety Car active status
        return (lap_bin, wear_bin, fuel_bin, gap_bin, weather_bin, self.sc_active)
        
    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool]:
        """
        Transition function.
        Actions:
            0: Push (Speed)
            1: Conserve (Save tyres)
            2: Pit (Refresh compound)
            3: Defend (Defense stance)
            4: Attack (ERS deploy, overtake attempt)
        """
        done = False
        reward = 0.0
        
        # Dynamic weather step simulation
        # 10% chance of dynamic dampness fluctuation
        if random.random() < 0.10:
            self.dampness = min(1.0, max(0.0, self.dampness + random.choice([-0.1, 0.1])))
            
        # Dynamic Safety Car draw
        self.sc_active = 1 if random.random() < 0.015 else 0
        
        # Action transitions & immediate base pacing rewards
        if action == 0:  # Push
            reward += 12.0
            self.tire_wear += 0.038
            self.fuel_level -= 2.2
            
        elif action == 1:  # Conserve
            reward += 6.5
            self.tire_wear += 0.010
            self.fuel_level -= 1.6
            
        elif action == 2:  # Pit
            pit_loss = 12.0 if self.sc_active == 1 else 22.0
            reward -= pit_loss
            self.tire_wear = 0.0
            
            # Additional penalty if pitting into traffic
            state = self._get_state()
            if state[3] == 1:
                reward -= 10.0
                
        elif action == 3:  # Defend
            # Slower pace but provides position holding security
            reward += 8.0
            self.tire_wear += 0.015
            self.fuel_level -= 1.8
            
        elif action == 4:  # Attack
            # High speed deploy ERS
            reward += 15.0
            self.tire_wear += 0.045
            self.fuel_level -= 2.6
            
            # Overtake bonus if rejoining near another car
            state = self._get_state()
            if state[3] == 1 and random.random() < 0.40:
                reward += 10.0 # Overtake success bonus
                
        # Tire wear penalties
        if self.tire_wear >= 1.0:
            reward -= 150.0  # blowout penalty
            done = True
        elif self.tire_wear > 0.45:
            reward -= 6.0  # wear degradation penalty
            
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
