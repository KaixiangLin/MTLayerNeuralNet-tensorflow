import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
import tensorflow as tf
import os
import MTLayer as models
import time
import sys
from scipy.signal import savgol_filter
from scipy import stats
from shutil import copyfile
from MTLayer_configure import FLAGS
import scipy.io as sio
current_time = FLAGS.current_time


# The DDI dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 1


'''
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
'''



def batch_data(x, y, batch_size):
    '''Input numpy array, and batch size'''
    data_size = len(x)
    index = np.random.permutation(data_size)
    # index = range(data_size)
    batch_index = index[:batch_size]

    batch_x = x[batch_index]
    batch_y = y[batch_index]
    return batch_x, batch_y

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.

    Returns:
    features_placeholder: features placeholder.
    labels_placeholder: Labels placeholder.
    """

    features_placeholder_t = tf.placeholder(tf.float32, shape=(batch_size,
                                                                FLAGS.FEATURE_SIZE))
    labels_placeholder_t = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.NUM_CLASS))
    dropout_placeholder_t = tf.placeholder(tf.float32)
    return features_placeholder_t, labels_placeholder_t, dropout_placeholder_t

def fill_feed_dict_run(train_x_t, train_y_t, dropout_rate_t,
                       features_placeholder_t, labels_placeholder_t, dropout_placeholder_t):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    batch_size_t = FLAGS.batch_size_t

    if batch_size_t < len(train_y_t):
        batch_x_t, batch_y_t = batch_data(train_x_t, train_y_t, batch_size_t)
    else:
        batch_x_t, batch_y_t = train_x_t, train_y_t


    batch_x_t = np.reshape(batch_x_t, (batch_size_t, FLAGS.FEATURE_SIZE))
    batch_y_t = np.reshape(batch_y_t, (batch_size_t, FLAGS.NUM_CLASS))


    feed_dict = {}

    feed_dict[features_placeholder_t] = batch_x_t
    feed_dict[labels_placeholder_t] = batch_y_t
    feed_dict[dropout_placeholder_t] = dropout_rate_t

    return feed_dict

def do_eval(sess, eval_correct_t,
            features_placeholder_t, labels_placeholder_t, dropout_placeholder_t,
            train_x_t, train_y_t,dropout_rate_t):
    """Runs one evaluation against the full epoch of data.

    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    features_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.

    num_examples_t = 0
    steps_per_epoch_t = 0
    batch_eavl = 0

    data_set_size = np.size(train_y_t)
    batch_size = FLAGS.batch_size_t

    if batch_size <= data_set_size:
        steps_per_epoch_t = data_set_size // batch_size
        num_examples_t = steps_per_epoch_t * batch_size
    else:
        num_repeat = batch_size // data_set_size  # + 1
        xtemp = train_x_t
        ytemp = train_y_t
        for i in range(num_repeat):
            xtemp = np.concatenate((xtemp, train_x_t), axis=0)
            ytemp = np.concatenate((ytemp, train_y_t), axis=0)

        train_x_t = xtemp[:batch_size]
        train_y_t = ytemp[:batch_size]
        steps_per_epoch_t = 1
        num_examples_t = batch_size

    for step in range(steps_per_epoch_t):
        feed_dict = fill_feed_dict_run(train_x_t, train_y_t, dropout_rate_t,
                                   features_placeholder_t, labels_placeholder_t, dropout_placeholder_t)

        batch_eavl += sess.run(eval_correct_t, feed_dict=feed_dict)

    batch_eavl = batch_eavl/num_examples_t

    return batch_eavl



def run_training(dataset_tuple):
    """Train DDI for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on DDI.


    train_x_t, train_y_t = dataset_tuple
    gradients = []

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        features_placeholder_t, labels_placeholder_t, dropout_placeholder_t = placeholder_inputs(FLAGS.batch_size_t)

        # Build a Graph that computes predictions from the inference model.
        placeholder_tuple = tuple([features_placeholder_t, labels_placeholder_t, dropout_placeholder_t])

        hidden_last, regularizers = models.inference(placeholder_tuple,
                                                     FLAGS.hidden_units,
                                                     FLAGS.FEATURE_SIZE)

        # Add to the Graph the Ops for loss calculation.
        loss = models.lossdef(hidden_last, labels_placeholder_t, regularizers, FLAGS.l2_reg_para)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, gradient_op = models.training(loss, FLAGS.learning_rate_t)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct_t = models.evaluation(hidden_last, labels_placeholder_t)

        # TODO debug

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        # Create a saver for first hidden layer variables.

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        obj_value_values = []


        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict_run(train_x_t, train_y_t, FLAGS.dropout_rate_t,
                                           features_placeholder_t, labels_placeholder_t, dropout_placeholder_t)


            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.

            _, loss_value, regularization_value, eval_correct_t_val, gradients = sess.run([train_op,
                                                                                loss,
                                                                                regularizers,
                                                                                eval_correct_t,
                                                                                gradient_op],
                                                                                feed_dict=feed_dict)


            obj_value_values.append(loss_value)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0 or step == 1:
                # Print status to stdout.
                print('Step %d: obj = %.2f reg = %.2f (%.3f sec)' % (step,
                                                                     loss_value,
                                                                     regularization_value,
                                                                     duration))
                # print(logits_value)
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)


                # Evaluate against the training set.
                print('Training Data Eval:')
                print('Step %d: training error  = %.2f ' % (step,
                                                            eval_correct_t_val))



        if FLAGS.print_grad:
            print 'Print gradient: \n', gradients, '\n'


        fig = plt.figure(1)
        plt.plot(obj_value_values)
        plt.savefig(FLAGS.train_dir + current_time + '_train_err.png')
        plt.close(fig)


def record_configure_files():
    copyfile('MTLayer_configure.py', FLAGS.train_dir + 'MTLayer_configure.txt')

def main(argv):
    ''' multi task dnn for arbitrary number of tasks
    '''
    # record print out info
    print('\n' + FLAGS.train_dir + '\n')

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if argv:
        FLAGS.L = argv[1]
    if len(argv) > 2:
        print('Usage(): argv1: Integer, the number of layers')

    '''record configure files'''
    record_configure_files()

    print FLAGS.L

    '''Load data'''
    fname = FLAGS.input_dir + FLAGS.input_filename_all  # with normalization
    mat_contents = sio.loadmat(fname)

    dataset_tuple = tuple([mat_contents['x'], mat_contents['y']])
    features, labels = dataset_tuple

    '''Normalize data'''
    features_normalized = stats.zscore(features, axis=0, ddof=1) # specified axis, using n-1 degrees of freedom (ddof=1) to calculate the standard deviation:
    dataset_tuple = tuple([features_normalized, labels])

    print('\n The current time is : %s \n' % current_time)
    run_training(dataset_tuple)


if __name__ == "__main__":
    tf.app.run()

