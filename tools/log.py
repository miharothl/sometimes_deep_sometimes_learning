from keras.callbacks import TensorBoard
import tensorflow as tf

def tf_log_start(model, timestamp):

    path = "./_logs/" + timestamp

    tensorboard = TensorBoard(log_dir=path, histogram_freq=0,
                              write_graph=True, write_images=False)
    tensorboard.set_model(model)
    return tensorboard

def tf_log(tensorboard, e, name, value):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    tensorboard.writer.add_summary(summary, e)
    tensorboard.writer.flush()