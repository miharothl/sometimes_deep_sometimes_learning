#First we import some libraries
#Json for loading and saving the model (optional)
import argparse
import json
import datetime
#matplotlib for rendering
# import matplotlib.pyplot as plt
#numpy for handeling matrix operations
import numpy as np
#time, to, well... keep track of time
import time
#Python image libarary for rendering
from PIL import Image
#iPython display for making sure we can render the frames
from IPython import display
#seaborn for rendering
# import seaborn

#Keras is a deep learning libarary
from keras.layers import *
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop

import numpy

#setting up seaborn
# seaborn.set()
from tools.environment import Catch
from tools.model import baseline_model_improved, models, save_model, load_model
from tools.log import tf_log_start, tf_log


class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory.
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, max_memory=100, discount=.9):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience

        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), 10,10,1))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        # targets = np.zeros((inputs.shape[0], num_actions))
        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i:i + 1] = state_t.reshape(1,10,10,1)

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t.reshape(1,10,10,1))[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1.reshape(1,10,10,1))[0])

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets



# parameters
epsilon = .15  # exploration
num_actions = 3  # [move_left, stay, move_right]
max_memory = 5000 # Maximum number of experiences we are storing
hidden_size = 200 # Size of the hidden layers
batch_size = 500 # Number of experiences we use for training per batch
grid_size = 10 # Size of the playing field



# Define environment/game
env = Catch(grid_size)

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)


def show_game(snapshots):
    rows = ([""]*len(snapshots))

    for s in range(len(snapshots)):
        for y in range(snapshots[s].shape[1]):
            rows[y] += "|"
            for x in range(snapshots[s].shape[0]):
                if snapshots[s][y][x]:
                    rows[y] = rows[y] + "*"
                else:
                    rows[y] = rows[y] + " "

    for row in rows:
        print(row)

def moving_average_diff(a, n=100):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def test(model):
    #This function lets a pretrained model play the game to evaluate how well it is doing
    global last_frame_time
    # plt.ion()
    # Define environment, game
    env = Catch(grid_size)
    #c is a simple counter variable keeping track of how much we train
    c = 0
    #Reset the last frame time (we are starting from 0)
    last_frame_time = 0
    #Reset score
    points = 0

    win = 0.

    #For training we are playing the game 10 times
    for e in range(100):

        loss = 0.
        #Reset the game
        env.reset()
        #The game is not over
        game_over = False
        # get initial input
        input_t = env.observe()

        game_play = []
        snapshot = numpy.array(input_t)
        game_play.append(snapshot.reshape(10, 10))

        #display_screen(3,points,input_t)
        c += 1
        while not game_over:
            #The learner is acting on the last observed game screen
            #input_t is a vector containing representing the game screen
            input_tm1 = input_t
            #Feed the learner the current status and get the expected rewards for different actions from it
            q = model.predict(input_tm1.reshape(1,10,10,1))
            #Select the action with the highest expected reward
            action = np.argmax(q[0])
            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            #Update our score
            points += reward
            c += 1

            if reward == 1:
                win = win + 1

            # display_screen(action,points,input_t)
            snapshot = numpy.array(input_t)
            game_play.append(snapshot.reshape(10, 10))

        show_game(game_play)

    return win/100


def train(model, epochs, tag, verbose=1):

    # Train
    # Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    # Epochs is the number of games we play

    best_accuracy= -1000.
    accuracy = 0.
    ma_score = 0.

    tensorboard = tf_log_start(model, tag)

    for e in range(epochs):
        loss = 0.
        # Resetting the game
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        game_play = []
        snapshot = numpy.array(input_t)
        game_play.append(snapshot.reshape(10, 10))

        while not game_over:
            # The learner is acting on the last observed game screen
            # input_t is a vector containing representing the game screen
            input_tm1 = input_t

            """
            We want to avoid that the learner settles on a local minimum.
            Imagine you are eating eating in an exotic restaurant. After some experimentation you find 
            that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
            food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
            It's simple: Sometimes, they just don't let you choose but order something random from the menu.
            Maybe you'll like it.
            The chance that your friends order for you is epsilon
            """
            if np.random.rand() <= epsilon:
                # Eat something random from the menu
                action = np.random.randint(0, num_actions, size=1)
            else:
                # Choose yourself
                # q contains the expected rewards for the actions
                q = model.predict(input_tm1.reshape(1,10,10,1))
                # We pick the action with the highest expected reward
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            snapshot = numpy.array(input_t)
            game_play.append(snapshot.reshape(10, 10))

            # If we managed to catch the fruit we add 1 to our win counter
            if reward == 1:
                win_cnt += 1

                # Uncomment this to render the game here
            # display_screen(action,3000,inputs[0])

            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # Load batch of experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # train model on experiences
            batch_loss = model.train_on_batch(inputs, targets)

            # print(loss)
            loss += batch_loss

        tf_log(tensorboard, e, "loss", loss)

        ma = moving_average_diff(win_hist)

        if len(ma) > 1:
            ma_score = ma[-1]
            tf_log(tensorboard, e, "moving average", ma_score)

        tf_log(tensorboard, e, "wins", win_cnt)

        if e % 5 == 4:

            accuracy = test(model)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            tf_log(tensorboard, e, "accuracy", accuracy)

        if e % 5 ==4:
            path = "./_models/" + tag + "-{:03}".format(e)
            save_model(model, path)


        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | Moving Avarage {} | Accuracy {} | Best Accuracy {}".format(e, epochs, loss, win_cnt, ma_score, accuracy, best_accuracy))

            show_game(game_play)
        win_hist.append(win_cnt)

    return win_hist

# epoch = 5000 # Number of games played in training, I found the model needs about 4,000 games till it plays well
# # Train the model
# # For simplicity of the noteb
# if 1:
#     hist = train(model, epoch, verbose=1)
#     print("Training done")
# # test(model)


def init(model_fn, tag, train, play):
    model = model_fn(grid_size, num_actions, hidden_size)
    model.summary()

    if play:
        model = load_model(play)
        test(model)

    if train:
        epoch = 5000 # Number of games played in training, I found the model needs about 4,000 games till it plays well
        hist = train(model, epoch, tag, verbose=1)
        print("Training done")


def main():
    ap = argparse.ArgumentParser(description="Pre-process adobe-raw file."
        , formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ap.add_argument('--train', dest='train', action='store_true', help="help me")
    # ap.add_argument('--no-train', dest='train', action='store_false')
    # ap.set_defaults(train=True)

    ap.add_argument('-m', '--model',
                    default="basic",
                    choices=models.keys())

    ap.add_argument('-t', '--train',
                    dest='train',
                    action='store_false',
                    help="Train switch")

    ap.add_argument('-p', '--play',
                    default="_models/improved-20180619T204302-059",
                    help="Train switch")

    # # ap.add_argument('-t', '--target_file_path',
    #                 default="/data/vf/adobe/raw/hit_data_preprocessed.tsv",
    #                 help="path to pre-processed hit_data.tsv file")
    # ap.add_argument('-i', '--source_ip_address',
    #                 default="127.0.0.1",
    #                 help="IP Address of a box running the script")
    # ap.add_argument('-d', '--target_delimiter',
    #                 default="\t",
    #                 help="Delimiter to use when creating target file")
    # ap.add_argument('-e', '--extension',
    #                 default=".gz.ok",
    #                 help="File extension of the control file")
    args = ap.parse_args()

    print(args.train)
    print(args.play)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    tag = args.model + "-" + timestamp

    if args.model:
         model_fn = models[args.model]
         init(model_fn, tag, train, args.play)


if __name__ == '__main__':
    main()

