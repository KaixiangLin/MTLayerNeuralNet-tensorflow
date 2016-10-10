from numpy.random import normal
import numpy as np
import scipy as sci
import sklearn as sk
import csv
from collections import Counter
from collections import defaultdict
import tensorflow as tf
import os
import random
import math
FLAGS = tf.app.flags.FLAGS



def inference(placeholder_tuple, hidden_units, FEATURE_SIZE, activation_func):
    """Build the multi layer neural network model up to where it may be used for inference.

          Args:
            placeholder_tuple: a list of placeholders, features_tasks[0] for first task.
            hidden_units: a list of hidden layer size [10, 10, 10] for three hidden layers, each has 10 nodes
            FEATURE_SIZE: size of features
          Returns:
            sigmoid: Output tensor with the computed logits.
    """

    features_tasks, labels_tasks, dropout_tasks = placeholder_tuple

    L = len(hidden_units)  # number of layers
    regularizers = 0     # regularization term
    hiddens = [None] * L   # hidden layers outputs

    weights = [None] * L
    biases = [None] * L
    for ll in range(L):

        if ll == 0:
            '''The first input layers'''
            # Hidden 1
            with tf.name_scope('hidden' + str(ll + 1)):
                weights[ll] = tf.Variable(
                    tf.truncated_normal([FEATURE_SIZE, hidden_units[ll]],
                                        stddev=1.0 / math.sqrt(float(FEATURE_SIZE))),
                    name='weights')
                biases[ll] = tf.Variable(tf.zeros([hidden_units[ll]]),
                                      name='biases')
                if activation_func == 1:
                    hiddens[ll] = tf.sigmoid(tf.matmul(features_tasks, weights[ll]) + biases[ll])  # task 1 output of layer 1
                else:
                    hiddens[ll] = tf.tanh(tf.matmul(features_tasks, weights[ll]) + biases[ll])  # task 1 output of layer 1

                regularizers += tf.nn.l2_loss(weights[ll]) + tf.nn.l2_loss(biases[ll])
                tf.histogram_summary("Layer" + str(ll + 1) + '/weights', weights[ll])

        elif ll >= 1:

            '''Intermediate layers from layer 2 to Layer L '''
            with tf.name_scope('hidden' + str(ll + 1)):
                weights[ll] = tf.Variable(
                    tf.truncated_normal([hidden_units[ll-1], hidden_units[ll]],
                                        stddev=1.0 / math.sqrt(hidden_units[ll-1])), name='weights')

                biases[ll] = tf.Variable(tf.zeros([hidden_units[ll]]), name='biases')

                if activation_func == 1:
                    hiddens[ll] = tf.sigmoid(tf.matmul(hiddens[ll-1], weights[ll]) + biases[ll])  # task 1 output of layer 1
                else:
                    hiddens[ll] = tf.tanh(tf.matmul(hiddens[ll-1], weights[ll]) + biases[ll])  # task 1 output of layer 1

                regularizers += tf.nn.l2_loss(weights[ll]) + tf.nn.l2_loss(biases[ll])
                tf.histogram_summary("Layer" + str(ll + 1) + '/weights', weights[ll])

    # hiddens[L-1] is the predicted outputs
    hidden_last = hiddens[L-1]
    return hidden_last, regularizers


def lossdef(hidden_last, labels, regularizers, l2_reg_para):
    ''' Define the loss function

    :param hidden_last: the predicted outputs
    :param labels:     the ground truth
    :param regularizers:
    :return: the loss function value
    '''


    loss = tf.nn.l2_loss(hidden_last - labels) + regularizers * l2_reg_para

    return loss


def training(loss, learning_rate):
    ''' Define training operator and choose different optimizer.

    :param loss:
    :param learning_rate:
    :return: train_op: training operations
    '''

    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradient_op = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(loss)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op, gradient_op

def evaluation(hidden_pred, labels_true):
    ''' define how to evaluate the performance

    :param hidden_pred:
    :param labels_true:
    :return: error
    '''

    error = tf.nn.l2_loss(hidden_pred - labels_true)

    return error