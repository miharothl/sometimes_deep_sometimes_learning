#Keras is a deep learning libarary
from keras.layers import *
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop

def baseline_model(grid_size,num_actions):
    #seting up the model with keras
    model = Sequential()
    model.add(Dense(100, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.1), "mse")
    return model


def baseline_model_improved(grid_size, num_actions):
    #seting up the model with keras

    nb_frames = 1
    model = Sequential()
    model.add(Conv2D(16, 3, 3, activation='relu', input_shape=(grid_size, grid_size, nb_frames)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(RMSprop(), 'MSE')
    return model

models = {'basic': baseline_model,
          'improved': baseline_model_improved}

def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + ".h5")
    print("Saved model to disk")

def load_model(path):
    # load json and create model
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + ".h5")
    print("Loaded model from disk")
    return loaded_model
