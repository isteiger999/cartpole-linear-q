import numpy as np
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode=None)
state, info = env.reset()
done = False
epsilon = 0.2
alpha = 0.00005  # learning rate
gamma = 0.95  # discount factor


def define_Q(Q, state, action):
        phi = np.array([state[0], state[1], state[2], state[3], np.sin(state[2]), np.cos(state[2]), state[0]**2, state[1]**2, state[2]**2, state[3]**2, state[0]*state[2], state[1]*state[3]]).reshape((12, 1))
        if action == 0:
            theta = Q["th1"]
            return float(phi.T @ theta)
        else:
            theta = Q["th2"]
            return float(phi.T @ theta)

def Q_learning_param(Q, state, action, next_state, reward):

    Q_plus = reward + gamma * max(define_Q(Q, next_state, 0), define_Q(Q, next_state, 1))
    phi = np.array([state[0], state[1], state[2], state[3], np.sin(state[2]), np.cos(state[2]), state[0]**2, state[1]**2, state[2]**2, state[3]**2, state[0]*state[2], state[1]*state[3]]).reshape((12, 1))

    if action == 1:
        theta_next = Q["th1"] - alpha * (float(Q["th1"].T@phi) - Q_plus) * phi
        Q["th1"] = theta_next #update theta
        Q["1"] = define_Q(Q, state, 1) #update Q value itself
    elif action == 0:
        theta_next = Q["th2"] - alpha * (float(Q["th2"].T@phi) - Q_plus) * phi
        Q["th2"] = theta_next
        Q["0"] = define_Q(Q, state, 0)

    
def epsilon_greedy(Q):
    random = np.random.rand(1)
    if random <= epsilon:
         if np.random.rand(1) <= 0.5:
              return 0
         else:
              return 1
    else:
         if Q["1"] >= Q["0"]:
            return 1
         else:
            return 0

         
     
# Initialisation    
th1, th2 = np.zeros((12,1)), np.zeros((12,1))  # parameter vectors for action 0 and 1
phi1 = np.array([state[0], state[1], state[2], state[3], np.sin(state[2]), np.cos(state[2]), state[0]**2, state[1]**2, state[2]**2, state[3]**2, state[0]*state[2], state[1]*state[3]]).reshape((12, 1))
phi2 = np.array([state[0], state[1], state[2], state[3], np.sin(state[2]), np.cos(state[2]), state[0]**2, state[1]**2, state[2]**2, state[3]**2, state[0]*state[2], state[1]*state[3]]).reshape((12, 1))
Q = {'1': float(phi1.T @ th1), '0': float(phi2.T @ th2), "th1": th1, "th2": th2}


def train_Q():

    Episodes = 1000

    for episode in range(Episodes):
        state, info = env.reset()
        done = False
        print(episode)

        while not done: # not done
            env.render()  # shows the cart-pole visually

            # sample an action
            action = epsilon_greedy(Q)

            next_state, reward, terminated, truncated, info = env.step(action)
            # Update Q-function
            if action == 0:
                Q_learning_param(Q, state, action, next_state, reward)
            elif action == 1:
                Q_learning_param(Q, state, action, next_state, reward) 

            done = terminated or truncated  # terminated: means state exceeded limit, terminated: means max time reached (but stable zone wasn't left)
            state = next_state
            print("State:", state, "Reward:", reward)

# Train Q-function

train_Q()

# Verify if Q-Learning worked:
env = gym.make("CartPole-v1", render_mode="human") # added rendering again

def greedy_action(Q, state):
    q0 = define_Q(Q, state, 0)
    q1 = define_Q(Q, state, 1)
    return 0 if q0 >= q1 else 1

episodes = 10
for ep in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = greedy_action(Q, state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep+1}: total reward = {total_reward}")

env.close()