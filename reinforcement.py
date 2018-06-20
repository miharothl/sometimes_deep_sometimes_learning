import argparse
import datetime

from tools.train import *


def init(model_fn, tag, train_flag, play_flag):

    # parameters
    epsilon = .15  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    max_memory = 5000  # Maximum number of experiences we are storing
    grid_size = 10  # Size of the playing field

    model = model_fn(grid_size, num_actions)
    model.summary()

    # Define environment/game
    env = Catch(grid_size)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    if play_flag:
        model = load_model(play_flag)
        test(model, grid_size)

    if train_flag:
        epoch = 5000 # Number of games played in training, I found the model needs about 4,000 games till it plays well
        batch_size = 500
        train(model, env, exp_replay, epoch, batch_size, epsilon, num_actions, tag, verbose=1)
        print("Training done")


def main():
    ap = argparse.ArgumentParser(description="Pre-process adobe-raw file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('-m', '--model',
                    default="basic",
                    choices=models.keys())

    ap.add_argument('-t', '--train',
                    dest='train',
                    action='store_true',
                    help="Train switch")

    ap.add_argument('-p', '--play',
                    default="_models/improved-20180619T204302-059",
                    help="Train switch")

    args = ap.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    tag = args.model + "-" + timestamp

    if args.model:
        model_fn = models[args.model]
        init(model_fn, tag, args.train, args.play)


if __name__ == '__main__':
    main()
