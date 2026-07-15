import numpy as np
import random

class F1Environment:
    """
    Formula 1 stint pacing and pit stop environment model.
    Models states, state transitions, and reward penalties for pacing decisions.
    """
    def __init__(self, data):
        self.data = data.copy()
        self.total_laps = len(data)
        self.reset()
        
    def reset(self):
        self.current_lap = 0
        self.tire_wear = 0.0
        return self._get_state()
        
    def _get_state(self):
        # 1. Discretize Lap progress
        pct = self.current_lap / max(self.total_laps, 1)
        if pct < 0.30:
            lap_bin = 0
        elif pct < 0.70:
            lap_bin = 1
        else:
            lap_bin = 2
            
        # 2. Discretize Tire Wear
        if self.tire_wear < 0.15:
            wear_bin = 0
        elif self.tire_wear < 0.40:
            wear_bin = 1
        else:
            wear_bin = 2
            
        # 3. Discretize Traffic indicators
        exit_gap = 30.0
        if 'ExitGap' in self.data.columns and 0 <= self.current_lap < self.total_laps:
            exit_gap = self.data.loc[self.current_lap, 'ExitGap']
        traffic_bin = 1 if exit_gap < 1.5 else 0
        
        return (lap_bin, wear_bin, traffic_bin)
        
    def step(self, action):
        """
        Transition function: Actions: 0=Push, 1=Save, 2=Pit.
        Returns: next_state, reward, done
        """
        done = False
        reward = 0.0
        
        if action == 0:  # Push
            # Speed reward
            reward += 10.0
            # Higher tyre degradation
            self.tire_wear += 0.035
            
        elif action == 1:  # Save
            # Conservaton reward
            reward += 7.0
            # Lower tire wear
            self.tire_wear += 0.012
            
        elif action == 2:  # Pit
            # Pit lane loss time penalty
            reward -= 22.0
            # Reset tire wear
            self.tire_wear = 0.0
            
            # Check traffic on release
            state = self._get_state()
            if state[2] == 1:
                reward -= 15.0  # Dirty air release penalty
        
        # Tire wear penalties
        if self.tire_wear >= 1.0:
            reward -= 100.0  # Catastrophic wear blowout
            done = True
        elif self.tire_wear > 0.40:
            reward -= 5.0  # Wear degradation speed loss
            
        # Lap advancement
        self.current_lap += 1
        if self.current_lap >= self.total_laps:
            done = True
            
        next_state = self._get_state()
        return next_state, reward, done

class QLearningAgent:
    """
    Q-Learning agent managing Q-table state-action policy mappings.
    """
    def __init__(self, actions=[0, 1, 2], lr=0.1, discount=0.95, epsilon=0.15):
        self.actions = actions
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
        
    def choose_action(self, state):
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_vals = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_vals)
            best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
            return random.choice(best_actions)
            
    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        
        # Temporal difference Bellman equation update
        new_q = old_q + self.lr * (reward + self.discount * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

def train_agent(env, agent, episodes=1000):
    """
    Trains the Q-learning agent on the F1 environment.
    """
    rewards_history = []
    epsilon_decay = 0.995
    
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
        
    # Calculate rolling average rewards
    rolling_rewards = []
    for i in range(len(rewards_history)):
        start = max(0, i - 50)
        rolling_rewards.append(float(np.mean(rewards_history[start:i+1])))
        
    return rewards_history, rolling_rewards
