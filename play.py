import gymnasium as gym
from ppo_torch import Agent



env = gym.make("Hopper-v5", render_mode="human")
agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
agent.load_models()
observation, info = env.reset()

episode_over = False
while not episode_over:
    action, _, _ =  agent.choose_action(observation) # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated

env.close()