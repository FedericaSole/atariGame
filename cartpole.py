import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000
MIN_EPSILON = 0.01
MAX_EPSILON = 1
EPSILON_DECAY = 0.95


class MyAgent:
    def __init__(self, state_size, action_size, is_done):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = MAX_EPSILON  # exploration rate
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.is_done = is_done

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        # Sequential() creates the foundation of the layers.
        model = Sequential()
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer with # of actions: 2 nodes (left, right)
        model.add(Dense(self.action_size, activation='linear'))
        # Create the model based on the information above
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon = self.epsilon * EPSILON_DECAY
        return self.epsilon

    def choose_action(self, state):
        rand_param = random.uniform(0, 1)
        if rand_param < self.epsilon:
            new_action = np.random.randint(0, self.action_size)
            return new_action
        else:
            prediction = self.model.predict(state)
            return np.argmax(prediction[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    is_done = False
    player = MyAgent(state_size, action_size, is_done)
    currentEpisodeNum = 0
    batch_size = 32
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    while currentEpisodeNum < EPISODES:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        player.epsilon = player.choose_epsilon()
        for time in range(500):
            env.render()
            action = player.choose_action(state)
            next_state, reward, is_done, info = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            player.remember(state, action, reward, next_state, is_done)
            state = next_state

            if is_done:
                print("episode: " + str(currentEpisodeNum) + "/" + str(EPISODES) + " totRew: " + str(
                    time) + " epsilon: " + str(player.epsilon))
                is_done = False
                break

            if len(player.memory) > batch_size:
                player.replay(batch_size)

        currentEpisodeNum += 1