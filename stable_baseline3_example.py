import gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make('CartPole-v1')

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model for 10,000 steps
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model
model = PPO.load("ppo_cartpole")

# Evaluate the model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
