import random
import gym
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from math import *
from collections import deque
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 1500000
MIN_EPSILON = 0.01
MAX_EPSILON = 1
EPSILON_DECAY = 0.995
RANDOM_STEPS = 1000
RANDOM_STEPS = 200


class MyAgent:
    def __init__(self, env, is_done):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95  # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon = MAX_EPSILON
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.epsilon = 0.5
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.is_done = is_done

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    '''
    def CNN(self):

        # functional API
        frames_input = keras.layers.Input(self.sp, name='frames')
        #print(str(frames_input))
        actions_input = keras.layers.Input((self.action_size, ), name='mask')
        #print(str(actions_input))
        # normalize the frames
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        # the cnn part follows this paper: Playing Atari with Deep Reinforcement Learning
        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        conv_1 = Conv2D(16, (8, 8), activation='relu', strides=(4, 4))(normalized)
        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = Conv2D(32, (4, 4), activation='relu', strides=(2, 2))(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(self.action_size)(hidden)
        #print("output: "+str(output.shape))
        # Finally, we multiply the output by the mask!
        # filtered_output = keras.layers.merge([output, actions_input], mode='mul')
        filtered_output = keras.layers.multiply([output, actions_input])
        #print("actions: "+str(actions_input.shape))
        #print("filtered: "+str(filtered_output.shape))
        model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))

        return model
    '''

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # after some random steps (epsilon=1 so the action is chosen randomly) it computes a new epsilon
    def choose_epsilon(self, currentEpisodeNum):
        if currentEpisodeNum < RANDOM_STEPS:
            return self.epsilon
        self.epsilon = self.epsilon * EPSILON_DECAY
        print("cambiando eps: " + str(self.epsilon))
        return self.epsilon

    def choose_action(self, state, currentEpisodeNum, currentFrame):
        #self.epsilon = self.choose_epsilon(currentEpisodeNum, currentFrame)
        rand_param = random.uniform(0, 1)
        if (rand_param < self.epsilon):
            new_action = np.random.randint(0, self.action_size)
            return new_action
        else:
            prediction = self.model.predict(state)
            return np.argmax(prediction[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if self.is_done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)


def main():
    env = gym.make('Pong-ram-v0')
    is_done = False
    player = MyAgent(env, is_done)
    batch_size = 32
    currentEpisodeNum = 0
    totReward = 0

    while currentEpisodeNum < EPISODES:
        currentFrame = 1
        frame = env.reset()
        player.epsilon = player.choose_epsilon(currentEpisodeNum)
        while not is_done:
            action = player.choose_action(frame, currentEpisodeNum, currentFrame)
            nextFrame, reward, is_done, info = env.step(action)
            reward = reward if not is_done else -10
            totReward += reward

            player.remember(frame, action, reward, nextFrame, is_done)
            frame = nextFrame

            currentFrame += 1

        if is_done:
            print("totRew: " + str(totReward))
            player.update_target_model()
            totReward = 0
            is_done = False

        if currentEpisodeNum % 100 == 0: print("eps: " + str(player.epsilon) + "episode: " + str(currentEpisodeNum))
        if len(player.memory) > batch_size:
            player.replay(batch_size)

        currentEpisodeNum += 1


main()