"""
    Tensorflow Utils, 
Hides direct tensorflow manipulation from DQNAgent
"""
import tensorflow as tf


def create_session():
    return tf.Session()


def initialize_all_variables(sess):
    sess.run(tf.global_variables_initializer())


def copy_variable_scope(source_name, target_name):
    """
        Copy network weights from one to another using variable_scope name, 
        returns copy_operation
    """
    update_target_expr = []
    source_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=source_name)
    target_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=target_name)
    for source_var, target_var in zip(source_vars, target_vars):
        #print(source_var, target_var)
        update_target_expr.append(target_var.assign(source_var))
    update_target_expr = tf.group(*update_target_expr)
    return update_target_expr


def create_saver():
    saver = tf.train.Saver(max_to_keep=5)
    return saver


def create_filewriter(logdir, graph=None):
    writer = tf.summary.FileWriter(logdir, graph)
    return writer


def create_summary_value(tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    return summary