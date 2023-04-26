import os
import imageio
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make ("MountainCar-v0", render_mode = "rgb_array")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20] * len (env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

RENDERS_DIRECTORY = "renders"
QTABLES_DIRECTORY = "qtables"

q_table = np.random.uniform (low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state (state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple (discrete_state.astype (int))

def create_directory (directory):
    if not os.path.exists(f"{directory}/"):
        os.makedirs(f"{directory}/")

create_directory (RENDERS_DIRECTORY)
create_directory (QTABLES_DIRECTORY)

for episode in range (EPISODES):
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        render = True
        print (episode)
    else:
        render = False

    discrete_state = get_discrete_state (env.reset ()[0])
    frames = []
    done = False

    while not done:
        if np.random.random () > epsilon:
            action = np.argmax (q_table[discrete_state])
        else:
            action = np.random.randint (0, env.action_space.n)

        new_state, reward, done, _, _ = env.step (action)
        
        # Punish agent at every timestep to encourage it to take actions that get to the goal faster
        reward -= 1

        episode_reward += reward
        new_discrete_state = get_discrete_state (new_state)

        if render:
            frames.append(env.render ())

        if not done:
            max_future_q = np.max (q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print (f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append (episode_reward)

    if not episode % 100:
        np.save (f"{QTABLES_DIRECTORY}/{episode}-qtable.npy", q_table)

    if not episode % SHOW_EVERY:
        average_reward = sum (ep_rewards[-SHOW_EVERY:]) / len (ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append (episode)
        aggr_ep_rewards['avg'].append (average_reward)
        aggr_ep_rewards['min'].append (min (ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append (max (ep_rewards[-SHOW_EVERY:]))

        print (f"Episode: {episode} avg: {average_reward} min: {min (ep_rewards[-SHOW_EVERY:])} max: {max (ep_rewards[-SHOW_EVERY:])}")

    if render:
        print(frames[0].shape)
        imageio.mimsave(f'{RENDERS_DIRECTORY}/{episode}.gif', frames, fps=40)

env.close ()

plt.plot (aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot (aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot (aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.legend (loc = 4)
plt.show ()
