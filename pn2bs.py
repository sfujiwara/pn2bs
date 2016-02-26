# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer


def inference(x_placeholder, n_in, n_hidden1, n_hidden2):
    """
    Description
    -----------
    Forward step which build graph.

    Parameters
    ----------
    x_placeholder: Placeholder for feature vectors
    n_in: Number of units in input layer which is dimension of feature
    n_hidden1: Number of units in hidden layer 1
    n_hidden2: Number of units in hidden layer 2

    Returns
    -------
    y_bs: Output tensor of predicted values for base stats
    y_type1: Output tensor of predicted values for type 1
    y_type2: Output tensor of predicted values for type 2
    """
    # Hidden1
    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([n_in, n_hidden1]),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([n_hidden1]))
        hidden1 = tf.nn.sigmoid(tf.matmul(x_placeholder, weights) + biases)

    # Hidden2
    with tf.name_scope('hidden2') as scope:
        weights = tf.Variable(
            tf.truncated_normal([n_hidden1, n_hidden2]),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([n_hidden2]))
        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)

    # Output layer for base stats
    with tf.name_scope('output_base_stats') as scope:
        weights = tf.Variable(
            tf.truncated_normal([n_hidden2, 6]),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([6]))
        y_bs = tf.matmul(hidden2, weights) + biases

    # Output layer for type1
    with tf.name_scope('output_type1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([n_hidden2, 18]),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([18]))
        # y_type1 = tf.nn.softmax(tf.matmul(hidden2, weights) + biases)
        y_type1 = tf.matmul(hidden2, weights) + biases

    # Output layer for type2
    with tf.name_scope('output_type2') as scope:
        weights = tf.Variable(
            tf.truncated_normal([n_hidden2, 19]),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([19]))
        y_type2 = tf.matmul(hidden2, weights) + biases
        # y_type2 = tf.nn.softmax(tf.matmul(hidden2, weights) + biases)

    return [y_bs, y_type1, y_type2]


def build_loss_bs(y_bs, t_ph_bs):
    """
    Parameters
    ----------
    y_bs: Output tensor of predicted values for base stats
    t_ph_bs: Placeholder for base stats

    Returns
    -------
    Loss tensor which includes placeholder of features and labels
    """
    loss_bs = tf.reduce_mean(tf.nn.l2_loss(t_ph_bs - y_bs), name='LossBaseStats')
    return loss_bs


def build_loss_type1(y_type1, t_ph_type1):
    """
    Parameters
    ----------
    y_type1: Output tensor of predicted values for base stats
    t_ph_type1: Placeholder for base stats

    Returns
    -------
    Loss tensor which includes placeholder of features and labels
    """
    loss_type1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_type1, t_ph_type1),
        name='LossType1'
    )
    return loss_type1


def build_loss_type2(y_type2, t_ph_type2):
    """
    Parameters
    ----------
    y_type2: Output tensor of predicted values for base stats
    t_ph_type2: Placeholder for base stats

    Returns
    -------
    Loss tensor which includes placeholder of features and labels
    """
    loss_type2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_type2, t_ph_type2),
        name='LossType2'
    )
    return loss_type2


def build_optimizer(loss, step_size):
    """
    Parameters
    ----------
    loss: Tensor of objective value to be minimized
    step_size: Step size for gradient descent

    Returns
    -------
    Operation of optimization
    """
    optimizer = tf.train.GradientDescentOptimizer(step_size)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


if __name__ == '__main__':
    # Set seed
    tf.set_random_seed(0)

    # Load data set and extract features
    df = pd.read_csv('data/poke_selected.csv')

    # Fill nulls in type2
    df.loc[df.type2.isnull(), 'type2'] = '無'

    # Vectorize pokemon name
    pokename_vectorizer = CountVectorizer(analyzer='char', min_df=1, ngram_range=(1, 2))
    x = pokename_vectorizer.fit_transform(list(df['name_jp'])).toarray()
    t_bs = np.array(df[['hp', 'attack', 'block', 'contact', 'defense', 'speed']])

    # Vectorize pokemon type
    # poketype_vectorizer = DictVectorizer(sparse=False)
    # d = df[['type1', 'type2']].to_dict('record')
    # t = np.hstack([bs, poketype_vectorizer.fit_transform(d)])

    # Vectorize pokemon type1
    poketype1_vectorizer = DictVectorizer(sparse=False)
    d = df[['type1']].to_dict('record')
    t_type1 = poketype1_vectorizer.fit_transform(d)

    # Vectorize pokemon type2
    poketype2_vectorizer = DictVectorizer(sparse=False)
    d = df[['type2']].to_dict('record')
    t_type2 = poketype2_vectorizer.fit_transform(d)

    # Placeholders
    x_ph = tf.placeholder(dtype=tf.float32)
    t_ph_bs = tf.placeholder(dtype=tf.float32)
    t_ph_type1 = tf.placeholder(dtype=tf.float32)
    t_ph_type2 = tf.placeholder(dtype=tf.float32)

    # build graph, loss, and optimizer
    y_bs, y_type1, y_type2 = inference(x_ph, n_in=1403, n_hidden1=512, n_hidden2=256)
    loss_bs = build_loss_bs(y_bs, t_ph_bs)
    loss_type1 = build_loss_type1(y_type1, t_ph_type1)
    loss_type2 = build_loss_type2(y_type2, t_ph_type2)
    # loss = tf.add_n([1e-4 * loss_bs, loss_type1, loss_type2], name='ObjectiveFunction')
    loss = tf.add_n([1e-4 * loss_bs, loss_type1, loss_type2], name='ObjectiveFunction')
    # loss = tf.add_n([loss_type1, loss_type2], name='ObjectiveFunction')
    optim = build_optimizer(loss, 1e-1)

    # Create session
    sess = tf.Session()

    # Initialize variables
    init = tf.initialize_all_variables()
    sess.run(init)

    # Create summary writer and saver
    summary_writer = tf.train.SummaryWriter('tflogs', graph_def=sess.graph_def)
    tf.scalar_summary(loss.op.name, loss)
    tf.scalar_summary(loss_bs.op.name, loss_bs)
    tf.scalar_summary(loss_type1.op.name, loss_type1)
    tf.scalar_summary(loss_type2.op.name, loss_type2)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()

    # Run optimization
    for i in range(1500):
        # Choose indices for mini batch update
        ind = np.random.choice(802, 802)
        batch_xs = x[ind]
        batch_ts_bs = t_bs[ind]
        batch_ts_type1 = t_type1[ind]
        batch_ts_type2 = t_type2[ind]
        # Create feed dict
        fd = {
            x_ph: batch_xs,
            t_ph_bs: batch_ts_bs,
            t_ph_type1: batch_ts_type1,
            t_ph_type2: batch_ts_type2
        }
        # Run optimizer and update variables
        sess.run(optim, feed_dict=fd)
        # Show information and write summary in every n steps
        if i % 100 == 99:
            # Show num of epoch
            print 'Epoch:', i + 1, 'Mini-Batch Loss:', sess.run(loss, feed_dict=fd)
            # sess.run(sess.graph.get_tensor_by_name('hidden1/weights:0'))
            # Write summary and save checkpoint
            summary_str = sess.run(summary_op, feed_dict=fd)
            summary_writer.add_summary(summary_str, i)
            name_model_file = 'model_lmd1e-4_epoch_' + str(i+1) + '.ckpt'
            save_path = saver.save(sess, 'model/tensorflow/'+name_model_file)
    else:
        name_model_file = 'model_lmd1e-4_epoch_' + str(i+1) + '.ckpt'
        save_path = saver.save(sess, 'model/tensorflow/'+name_model_file)

    poke_name = 'サンダー'
    v = pokename_vectorizer.transform([poke_name]).toarray()
    pred_bs = sess.run(y_bs, feed_dict={x_ph: v})
    pred_type1 = np.argmax(sess.run(y_type1, feed_dict={x_ph: v}))
    pred_type2 = np.argmax(sess.run(y_type2, feed_dict={x_ph: v}))
    print poke_name
    print pred_bs
    print pred_type1, pred_type2
    print poketype1_vectorizer.get_feature_names()[pred_type1]
    print poketype2_vectorizer.get_feature_names()[pred_type2]

    # Save variables of TensorFlow
    # save_path = saver.save(sess, 'model/base-stats.ckpt')
    # Save vectorizer of scikit-learn
    joblib.dump(pokename_vectorizer, 'model/sklearn/pokemon-name-vectorizer')
    joblib.dump(poketype1_vectorizer, 'model/sklearn/pokemon-type1-vectorizer')
    joblib.dump(poketype2_vectorizer, 'model/sklearn/pokemon-type2-vectorizer')

    # I don't know how to save and load graph_def...
    # tf.train.write_graph(sess.graph_def, 'graph', 'pn2bs-graph.txt')
    # f = open('graph/pn2bs-graph.txt', mode='rb')
    # fileContent = f.read()
    # graph_def = tf.GraphDef()
    # graph_def.ParseFromString(fileContent)
