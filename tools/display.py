import time

#matplotlib for rendering
import matplotlib.pyplot as plt

from PIL import Image
#iPython display for making sure we can render the frames
from IPython import display
#seaborn for rendering
# import seaborn
import numpy as np


"""
Here we define some variables used for the game and rendering later
"""
#last frame time keeps track of which frame we are at
last_frame_time = 0
#translate the actions to human readable words
translate_action = ["Left","Stay","Right","Create Ball","End Test"]
#size of the game field
grid_size = 10

def set_max_fps(last_frame_time, FPS=1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1. / FPS - (current_milli_time() - last_frame_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()

def display_screen(action, points, input_t):
    # Function used to render the game screen
    # Get the last rendered frame
    global last_frame_time
    print("Action %s, Points: %d" % (translate_action[action], points))
    # Only display the game screen if the game is not over
    if ("End" not in translate_action[action]):
        # Render the game with matplotlib
        plt.imshow(input_t.reshape((grid_size,) * 2),
                   interpolation='none', cmap='gray')
        # Clear whatever we rendered before
        display.clear_output(wait=True)
        # And display the rendering
        display.display(plt.gcf())
    # Update the last frame time
    last_frame_time = set_max_fps(last_frame_time)


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

