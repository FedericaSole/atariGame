import random
import gym
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
import numpy as np
from math import *
from collections import deque
from keras.optimizers import Adam
from keras import backend as K


ATARI_SHAPE = (105, 80, 4)
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
MAX_EPISODES = 200000
EPISODES = 5000
MAX_FRAMES = 15000000
MIN_EPSILON = 0.01
MAX_EPSILON = 1
EPSILON_DECAY = 0.995
RANDOM_STEPS = 1000
N_ACTIONS = 6       #this is the number of actions specifically for the breakout game


class MyAgent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.CNN()
        self.target_model = self.CNN()
        self.update_target_model()
        self.is_done = False
        
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def CNN(self):

        # functional API
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
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


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #after some random steps (epsilon=1 so the action is chosen randomly) it computes a new epsilon using a sinusoidal decaying function
    def choose_epsilon(self, epsilon, currentEpisodeNum):
        if currentEpisodeNum < RANDOM_STEPS:
            return epsilon
        new_epsilon = epsilon * pow(EPSILON_DECAY, currentEpisodeNum) * 1 / 2 * (
                    1 + cos((2 * pi * currentEpisodeNum * 1) / MAX_EPISODES))   #1(one) instead of miniepochs num
        return new_epsilon

    def choose_action(self, state, epsilon, currentEpisodeNum):
        epsilon = self.choose_epsilon(epsilon, currentEpisodeNum)
        rand_param = random.uniform(0, 1)
        if (rand_param < epsilon):
            new_action = np.random.randint(0, N_ACTIONS)
            return new_action
        else:
            model = self.CNN()
            prediction = model.predict([state, np.ones(self.action_size)])
            return np.argmax(prediction[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            #print(str(state)+str(np.ones(self.action_size)))
            target = self.model.predict([state, np.ones(self.action_size)])
            if self.is_done:
                target[0][action] = reward
            else:
                t = self.target_model.predict([next_state, np.ones(self.action_size)])[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)


# resize the img frame (new_size = old_size/2) and make it black and white (scale of grey)
def preprocessing(img):
    new_img = img[::2, ::2, :]  # new img size = (105,80)
    return new_img[:, :, 0]


def main():
    env = gym.make('PongDeterministic-v4')
    player = MyAgent(env)
    #print(player.model.summary())
    old_frame = env.reset()
    frame = preprocessing(old_frame)
    batch_size = 32
    epsilon = MAX_EPSILON
    currentEpisodeNum = 0
    totReward = 0
    
    #while currentEpisodeNum < MAX_EPISODES:
    while currentEpisodeNum < 20000:
        #for currentFrame in range(0, 200000):
        for currentFrame in range(0, 10000):
            action = player.choose_action(frame, epsilon, currentEpisodeNum)

            nextFrame, reward, is_done, info = env.step(action)
            reward = reward if not is_done else -10
            totReward += reward

            next_processed_state = preprocessing(nextFrame)

            player.remember(frame, action, reward, next_processed_state, is_done)
            frame = preprocessing(nextFrame)

            if player.is_done:
                player.update_target_model()
                print(totReward)
                totReward = 0
                player.is_done = False
                frame = env.reset()
                break

        if len(player.memory) > batch_size:
                player.replay(batch_size)
        
        currentEpisodeNum+=1

main()

