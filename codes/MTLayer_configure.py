import time
import tensorflow as tf


current_time = time.strftime("%Y%m%d_%H-%M")
tf.app.flags.DEFINE_string('current_time', current_time, '')
tf.app.flags.DEFINE_string('train_dir', '../data/' + current_time + '/', 'Directory to put the check points data')
tf.app.flags.DEFINE_string('load_dir', '../data/', 'Directory to put the check points data')
tf.app.flags.DEFINE_string('input_dir', '../data/', 'Directory to download data files')

tf.app.flags.DEFINE_string('input_filename_all', 'problem-4.mat', 'input file for all tasks, features and labels')


tf.app.flags.DEFINE_string('FEATURE_SIZE', 10,'num of features')
tf.app.flags.DEFINE_integer('max_steps', int(5e+3), 'Number of steps to run trainer.')
tf.app.flags.DEFINE_integer('NUM_CLASS', 10, 'Number of class in labels')
tf.app.flags.DEFINE_integer('L', 3, 'Number of layers')
tf.app.flags.DEFINE_float('learning_rate_t', 1e-3, 'learning rate for first hidden layers')
tf.app.flags.DEFINE_float('l2_reg_para', 1e-3, 'regularization parameter ')
tf.app.flags.DEFINE_bool('print_grad', False, 'rate for training dataset')
FLAGS = tf.app.flags.FLAGS

# TODO debug
'''Layers specific parameters'''
tf.app.flags.DEFINE_integer('hidden_units', [10] * FLAGS.L, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('batch_size_t', 100, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_float('dropout_rate_t', 0.5, 'Initial the probability of keep the value')