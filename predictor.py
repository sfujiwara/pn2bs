# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
import pn2bs


# Placeholder
x_ph = tf.placeholder(dtype=tf.float32)
t_ph = tf.placeholder(dtype=tf.float32)
# Build graph
y_bs, y_type1, y_type2 = pn2bs.inference(x_ph, n_in=1403, n_hidden1=512, n_hidden2=256)
# Create session
sess = tf.Session()
# Load TensorFlow model
saver = tf.train.Saver()
saver.restore(sess, "model/tensorflow/model_lmd1e-4_epoch_1500.ckpt")
# Load vectorizer of scikit-learn
pokename_vectorizer = joblib.load("model/sklearn/pokemon-name-vectorizer")
poketype1_vectorizer = joblib.load("model/sklearn/pokemon-type1-vectorizer")
poketype2_vectorizer = joblib.load("model/sklearn/pokemon-type2-vectorizer")


def predict(poke_name):
    v = pokename_vectorizer.transform([poke_name]).toarray()
    pred_bs = sess.run(y_bs, feed_dict={x_ph: v})
    pred_type1 = np.argmax(sess.run(y_type1, feed_dict={x_ph: v}))
    pred_type2 = np.argmax(sess.run(y_type2, feed_dict={x_ph: v}))
    result = {
        'name': poke_name,
        'hp': float(pred_bs[0][0]),
        'attack': float(pred_bs[0][1]),
        'block': float(pred_bs[0][2]),
        'contact': float(pred_bs[0][3]),
        'defense': float(pred_bs[0][4]),
        'speed': float(pred_bs[0][5]),
        'type1': poketype1_vectorizer.get_feature_names()[pred_type1].split('=')[1],
        'type2': poketype2_vectorizer.get_feature_names()[pred_type2].split('=')[1],
    }
    return result


if __name__ == '__main__':
    result = predict('テンソルフロー')
    print result['name']
    print result['hp']
    print result['attack']
    print result['block']
    print result['contact']
    print result['defense']
    print result['speed']
    print result['type1']
    print result['type2']
