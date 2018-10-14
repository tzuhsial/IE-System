"""
    Tensorflow implementation of a fully connected network for baseline DQN
"""
import tensorflow as tf


class QNetwork(object):
    """
    2 Layer feed forward network
    """

    def __init__(self, opt, name='qnetwork'):
        # Network Architecture Hyperparameters
        self.input_size = int(opt['input_size'])
        self.hidden_size = int(opt['hidden_size'])
        self.output_size = int(opt['output_size'])

        # Unused for now
        # self.dropout_rate = opt['dropout_rate']

        # Build Graph, and training operation
        self._build_network(name)
        self._build_loss(name)

        optimizer_name = opt['optimizer']
        lr = float(opt['learning_rate'])

        if optimizer_name == "AdamOptimizer":
            self.optimizer = tf.train.AdamOptimizer(lr)  # Adam
        else:
            raise NotImplementedError("Optimizer: {}".format(optimizer_name))

        # Define graph running ops
        self.train_op = self.optimizer.minimize(self.loss)

    def _build_network(self, name):
        """
        Build network with name_scope
        """
        with tf.variable_scope(name):
            # Placeholders
            self.state_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, self.input_size))
            self.action_placeholder = tf.placeholder(
                dtype=tf.int32, shape=(None, ))
            self.qvalue_placeholder = tf.placeholder(
                dtype=tf.int32, shape=(None, ))

            # [ 4 * 100 * 2 ]
            self.hidden = tf.layers.dense(
                self.state_placeholder,
                self.hidden_size,
                activation=tf.nn.relu)
            self.all_qvalues_output = tf.layers.dense(
                self.hidden, self.output_size)  # (batch_size, output_size)

            # Gather with action indices
            mask = tf.one_hot(self.action_placeholder, depth=self.output_size)
            self.qvalues_output = tf.boolean_mask(self.all_qvalues_output,
                                                  mask)

    def _build_loss(self, name):
        self.loss = tf.losses.mean_squared_error(
            labels=self.qvalue_placeholder, predictions=self.qvalues_output)

    def train_batch(self, sess, batch_states, batch_actions,
                    batch_target_qvalues):
        # Create feed_dict
        feed_dict = {
            self.state_placeholder: batch_states,
            self.action_placeholder: batch_actions,
            self.qvalue_placeholder: batch_target_qvalues
        }
        # Create fetches
        fetches = [self.train_op, self.loss]
        # Run
        _, batch_loss = sess.run(fetches, feed_dict=feed_dict)
        return batch_loss

    def predict_batch(self, sess, batch_states):
        # Create feed_dict
        feed_dict = {self.state_placeholder: batch_states}
        # Create fetches
        fetches = [self.all_qvalues_output]
        # Run session
        batch_qvalues = sess.run(fetches, feed_dict=feed_dict)[0]
        return batch_qvalues


class LinearNetwork(QNetwork):
    def __init__(self, opt, name='qnetwork'):
        # Network Architecture Hyperparameters
        self.input_size = int(opt['input_size'])
        self.output_size = int(opt['output_size'])

        # Build Graph, and training operation
        self._build_network(name)
        self._build_loss(name)

        optimizer_name = opt['optimizer']
        lr = float(opt['learning_rate'])

        if optimizer_name == "AdamOptimizer":
            self.optimizer = tf.train.AdamOptimizer(lr)  # Adam
        else:
            raise NotImplementedError("Optimizer: {}".format(optimizer_name))

        # Define graph running ops
        self.train_op = self.optimizer.minimize(self.loss)

    def _build_network(self, name):
        """
        Build network with name_scope
        """
        with tf.variable_scope(name):
            # Placeholders
            self.state_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, self.input_size))
            self.action_placeholder = tf.placeholder(
                dtype=tf.int32, shape=(None, ))
            self.qvalue_placeholder = tf.placeholder(
                dtype=tf.int32, shape=(None, ))

            self.all_qvalues_output = tf.layers.dense(
                self.state_placeholder,
                self.output_size)  # (batch_size, output_size)

            # Gather with action indices
            mask = tf.one_hot(self.action_placeholder, depth=self.output_size)
            self.qvalues_output = tf.boolean_mask(self.all_qvalues_output,
                                                  mask)


class Critic(object):
    """
    2 Layer feed forward network, estimates value function V(S)
    """

    def __init__(self, opt, name='qnetwork'):
        # Network Architecture Hyperparameters
        self.input_size = int(opt['input_size'])
        self.hidden_size = int(opt['hidden_size'])
        self.output_size = int(opt['output_size'])

        assert self.output_size == 1, "Critic output size should be 1"
        # Unused for now
        # self.dropout_rate = opt['dropout_rate']

        # Build Graph, and training operation
        self._build_network(name)
        self._build_loss(name)

        optimizer_name = opt['optimizer']
        lr = float(opt['learning_rate'])

        if optimizer_name == "AdamOptimizer":
            self.optimizer = tf.train.AdamOptimizer(lr)  # Adam
        else:
            raise NotImplementedError("Optimizer: {}".format(optimizer_name))

        # Define graph running ops
        self.train_op = self.optimizer.minimize(self.loss)

    def _build_network(self, name):
        """
        Build network with name_scope
        """
        with tf.variable_scope(name):
            # Placeholders
            self.state_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, self.input_size))
            self.state_value_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, self.output_size))

            # [ 4 * 100 * 2 ]
            self.hidden = tf.layers.dense(
                self.state_placeholder,
                self.hidden_size,
                activation=tf.nn.relu)

            self.state_value_output = tf.layers.dense(
                self.hidden, self.output_size)  # (batch_size, output_size)

    def _build_loss(self, name):
        self.loss = tf.losses.mean_squared_error(
            labels=self.state_value_placeholder,
            predictions=self.state_value_output)

    def train_batch(self, sess, batch_states, batch_state_values):
        # Create feed_dict
        feed_dict = {
            self.state_placeholder: batch_states,
            self.state_value_placeholder: batch_state_values
        }
        # Create fetches
        fetches = [self.train_op, self.loss]
        # Run
        _, batch_loss = sess.run(fetches, feed_dict=feed_dict)
        return batch_loss

    def predict_batch(self, sess, batch_states):
        # Create feed_dict
        feed_dict = {self.state_placeholder: batch_states}
        # Create fetches
        fetches = [self.state_value_output]
        # Run session
        batch_state_values = sess.run(fetches, feed_dict=feed_dict)[0]
        return batch_state_values


class Actor(QNetwork):
    """
    2 layer feed forward network with softmax output
    """

    def __init__(self, opt, name='actor'):
        super(Actor, self).__init__(opt, name)

    def _build_network(self, name):
        """
        Build network with name_scope
        """
        with tf.variable_scope(name):
            # Placeholders
            self.state_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, self.input_size))
            self.action_placeholder = tf.placeholder(
                dtype=tf.int32, shape=(None))
            self.gt_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, 1))

            self.hidden = tf.layers.dense(
                self.state_placeholder,
                self.hidden_size,
                activation=tf.nn.relu)

            self.action_scores = tf.layers.dense(self.hidden, self.output_size)
            self.action_probs = tf.nn.softmax(self.action_scores)

    def _build_loss(self, name):
        one_hot_actions = tf.one_hot(
            self.action_placeholder, depth=self.output_size)
        update_action_prob = tf.reduce_sum(
            one_hot_actions * self.action_probs, axis=1)
        # Loss function
        tensor_loss = -self.gt_placeholder * tf.log(update_action_prob)
        self.loss = tf.reduce_sum(tensor_loss)

    def train_batch(self, sess, batch_states, batch_actions, batch_Gts):
        # Create feed_dict
        feed_dict = {
            self.state_placeholder: batch_states,
            self.action_placeholder: batch_actions,
            self.gt_placeholder: batch_Gts
        }
        # Create fetches
        fetches = [self.train_op, self.loss]
        # Run
        _, batch_loss = sess.run(fetches, feed_dict=feed_dict)
        return batch_loss

    def predict_batch(self, sess, batch_states):
        # Create feed_dict
        feed_dict = {self.state_placeholder: batch_states}
        # Create fetches
        fetches = [self.action_probs]
        # Run session
        action_probs = sess.run(fetches, feed_dict=feed_dict)[0]
        return action_probs
