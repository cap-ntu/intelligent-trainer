"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class NNValueFunction(object):
    """ NN-based state-value function """

    def __init__(self, obs_dim, hid1_mult, name_scope):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.name_scope = name_scope + '/trpo_val_func'
        self.lr = None  # learning rate set in _build_graph()
        self._build_graph()

    @property
    def sess(self):
        return tf.get_default_session()

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        with tf.variable_scope('trpo_val_func'):
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.var_list = []
        for var in var_list:
            if self.name_scope in var.name:
                self.var_list.append(var)
        self.init_op = tf.variables_initializer(var_list=self.var_list)


    def fit(self, x, y, logger=None):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat) / np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))  # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        # logger.log({'ValFuncLoss': loss,
        #             'ExplainedVarNew': exp_var,
        #             'ExplainedVarOld': old_exp_var})
        return loss, exp_var, old_exp_var

    def predict(self, x, step=None):
        """ Predict method """
        if step is not None:
            x = np.concatenate((x, np.reshape(np.array([step]), newshape=[-1])))
        if len(x.shape) < 2:
            x = np.reshape(x, newshape=[1, -1])
        elif len(x.shape) > 2:
            raise ValueError('Wrong dim of x')
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def init(self):
        self.sess.run(self.init_op)

    def copy_weight(self, new_val):
        t_change = []
        assert len(new_val.var_list) == len(self.var_list)
        for i in range(len(new_val.var_list)):
            t_change.append(-new_val.var_list[i] + self.var_list[i])
        grads_t = zip(t_change, self.var_list)
        target_update = tf.train.GradientDescentOptimizer(1.0).apply_gradients(grads_t)
        self.sess.run(target_update)
