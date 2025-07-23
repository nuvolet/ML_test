'''
Reinforcement Learning: Adaptive Weather Sampling

Problem:
Optimizing the deployment of weather observation systems (like weather balloons or drones) to maximize forecast improvement.

Real-World Application:
NASA's Global Modeling and Assimilation Office uses adaptive sampling techniques to optimize where to deploy additional atmospheric observations
(e.g., dropsondes from hurricane hunter aircraft). These RL approaches have been shown to improve hurricane track forecasts by strategically collecting data 
in regions where forecast uncertainty is highest.

'''


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create a weather sampling environment
class AdaptiveWeatherSamplingEnv(gym.Env):
    def __init__(self, weather_data_path, domain_size=(50, 50), n_drones=5, episode_length=24):
        super().__init__()
        
        # Load forecast ensemble data
        self.weather_data = xr.open_dataset(weather_data_path)
        self.ensemble_members = len(self.weather_data.ensemble)
        
        # Environment parameters
        self.domain_size = domain_size
        self.n_drones = n_drones
        self.episode_length = episode_length
        self.current_step = 0
        self.current_episode = 0
        
        # Action space: where to place each drone (normalized grid coordinates)
        self.action_space = spaces.Box(low=0, high=1, 
                                      shape=(self.n_drones, 2),  # x,y coordinates for each drone
                                      dtype=np.float32)
        
        # Observation space: current ensemble spread + previous drone positions
        self.observation_space = spaces.Dict({
            'ensemble_spread': spaces.Box(low=0, high=np.inf, shape=self.domain_size, dtype=np.float32),
            'previous_samples': spaces.Box(low=0, high=1, shape=(self.n_drones, 2), dtype=np.float32)
        })
        
        # Initialize state
        self.reset()
    
    def _get_ensemble_spread(self):
        """Get current ensemble spread map (uncertainty in forecasts)"""
        timestep_data = self.weather_data.isel(time=self.current_step)
        # Calculate standard deviation across ensemble members
        spread = timestep_data['temperature'].std(dim='ensemble').values
        # Normalize spread
        normalized_spread = (spread - spread.min()) / (spread.max() - spread.min() + 1e-6)
        return normalized_spread
    
    def _get_observation(self):
        """Construct the current observation"""
        return {
            'ensemble_spread': self._get_ensemble_spread(),
            'previous_samples': self.drone_positions
        }
    
    def _compute_reward(self, new_drone_positions):
        """
        Compute reward based on:
        1. Information gain (reduction in ensemble spread)
        2. Coverage of high-uncertainty regions
        3. Energy efficiency (penalize excessive movement)
        """
        # Get current ensemble spread
        spread = self._get_ensemble_spread()
        
        # Calculate information gain at sampling locations
        information_gain = 0
        for pos in new_drone_positions:
            # Convert normalized position to grid indices
            x, y = int(pos[0] * (self.domain_size[0] - 1)), int(pos[1] * (self.domain_size[1] - 1))
            # Higher reward for sampling high-uncertainty regions
            information_gain += spread[x, y]
        
        # Penalize duplicate sampling (drones too close to each other)
        positions_array = np.array(new_drone_positions)
        duplicates_penalty = 0
        for i in range(self.n_drones):
            for j in range(i+1, self.n_drones):
                # Calculate Euclidean distance between drones
                distance = np.linalg.norm(positions_array[i] - positions_array[j])
                if distance < 0.1:  # If drones are close
                    duplicates_penalty += 1
        
        # Penalize movement cost (if not first step)
        movement_cost = 0
        if hasattr(self, 'drone_positions'):
            for i in range(self.n_drones):
                # Calculate distance moved
                old_pos = self.drone_positions[i]
                new_pos = new_drone_positions[i]
                movement_cost += np.linalg.norm(new_pos - old_pos)
        
        # Total reward
        reward = (
            2.0 * information_gain -          # Reward for sampling high-uncertainty areas
            0.5 * duplicates_penalty -        # Penalty for duplicate sampling
            0.2 * movement_cost               # Penalty for excessive movement
        )
        
        return reward
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.current_step = 0
        self.current_episode += 1
        
        # Select a random starting day for this episode
        max_start = len(self.weather_data.time) - self.episode_length
        self.episode_start = np.random.randint(0, max_start)
        
        # Initialize drone positions randomly
        self.drone_positions = np.random.random((self.n_drones, 2)).astype(np.float32)
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment"""
        # Ensure action is in correct format
        assert action.shape == (self.n_drones, 2), f"Expected action shape {(self.n_drones, 2)}, got {action.shape}"
        
        # Clip actions to valid range [0, 1]
        drone_positions = np.clip(action, 0, 1)
        
        # Compute reward
        reward = self._compute_reward(drone_positions)
        
        # Update drone positions
        self.drone_positions = drone_positions
        
        # Update time step
        self.current_step += 1
        done = (self.current_step >= self.episode_length)
        
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        """Visualize the current state"""
        if mode != 'human':
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot ensemble spread (uncertainty)
        spread = self._get_ensemble_spread()
        plt.imshow(spread, cmap='viridis', origin='lower')
        plt.colorbar(label='Forecast Uncertainty')
        
        # Plot drone positions
        positions = self.drone_positions
        plt.scatter(
            positions[:, 0] * (self.domain_size[0] - 1),
            positions[:, 1] * (self.domain_size[1] - 1),
            c='red', marker='x', s=100, label='Drone Sampling Locations'
        )
        
        plt.title(f'Adaptive Weather Sampling - Step {self.current_step}')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Create the environment
env = AdaptiveWeatherSamplingEnv(weather_data_path='/path/to/ensemble_forecast.nc')
env = DummyVecEnv([lambda: env])

# Create and train a PPO agent
model = PPO("MultiInputPolicy", env, verbose=1, 
           learning_rate=0.0003,
           n_steps=2048,
           batch_size=64,
           n_epochs=10,
           gamma=0.99,
           tensorboard_log="./adaptive_sampling_tensorboard/")

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("adaptive_sampling_ppo")

# Evaluate the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()

# Compare with baseline strategies
def evaluate_strategy(env, strategy, n_episodes=10):
    total_rewards = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if strategy == 'random':
                action = env.action_space.sample()
            elif strategy == 'fixed_grid':
                # Place drones in a fixed grid pattern
                n = int(np.sqrt(env.n_drones))
                positions = []
                for i in range(n):
                    for j in range(n):
                        positions.append([i/n + 0.5/n, j/n + 0.5/n])
                positions = positions[:env.n_drones]  # Ensure correct number of drones
                action = np.array(positions)
            elif strategy == 'rl':
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    
    return total_rewards / n_episodes

# Evaluate different strategies
strategies = ['random', 'fixed_grid', 'rl']
results = {}

for strategy in strategies:
    avg_reward = evaluate_strategy(env, strategy, n_episodes=10)
    results[strategy] = avg_reward
    print(f"{strategy.upper()} strategy average reward: {avg_reward:.2f}")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Performance Comparison of Sampling Strategies')
plt.ylabel('Average Reward')
plt.ylim(bottom=0)
plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()