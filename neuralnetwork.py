import tensorflow as tf

import logging
import sys
from datetime import datetime
from time import time
import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import keras.backend as K
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold

from Ann_utilities import fit_model, plot_epoch_history
skf = StratifiedKFold(n_splits = 10, shuffle=True)


class BasicNN():
    @staticmethod
    def train_model(model, train_data_x, train_data_y):
        """
        Trains the model object using train_data_x and train_data_y
        """
        log = logging.getLogger('model.train')
        start_time = float(time())
        log.info('Start training model')
        print(model.summary())
        history = Ann_utilities.fit_model(model, train_data_x,train_data_y)
        log.info('Model trained in %fs' % (float(time()) - start_time))
        return model, history

    @staticmethod
    def test_model(model, test_data_x, test_data_y):
        log = logging.getLogger('model.test')
        log.info('Start testing model')
        start_time = float(time())
        [loss, metric] = model.evaluate(test_data_x, test_data_y, verbose=1)
        log.info("Loss: %f" % loss)
        log.info("Testing set Mean Abs Error / accuracy metric: %f" % (metric))
        test_prediction_probas = model.predict(test_data_x)
        test_predicted_labels = [np.argmax(x) for x in test_prediction_probas]
        return loss, metric, test_prediction_probas, test_predicted_labels
    @staticmethod
    def evaluate_model(model, train_x, train_y, test_x, test_y):
        log = logging.getLogger('model.evaluate')
        log.info('model.evaluate')
        model, train_history = train_model(model, train_x, train_y)
        Ann_utilities.plot_epoch_history(train_history, y_label = "Arbitrary Units", title = 'EpochHistory.png')
        test_loss, test_mae, test_preds_probs, test_preds = test_model(model, test_x, test_y)
        data = {
            'test_loss':test_loss,
            'test_mae':test_mae,
            'test_predictions':test_preds,
            'test_actual':test_y
        }
        for i in range(len(test_preds_probs[0])):
            data['test_probability_%s' % str(i+1)] = [x[i] for x in test_preds_probs]
        return pd.DataFrame(data=data), test_preds_probs

class Loss():
    @staticmethod
    def decov_loss(xs, name='decov_loss'):
        """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
        'Reducing Overfitting In Deep Networks by Decorrelating Representation'

        Args:
            xs: 4-D `tensor` [batch_size, height, width, channels], input

        Returns:
            a `float` decov loss
        """
        with tf.name_scope(name):
            x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
            m = tf.reduce_mean(x, 0, True)
            z = tf.expand_dims(x - m, 2)
            corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1])), 0)
            corr_frob_sqr = tf.reduce_sum(tf.square(corr))
            corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
            loss = 0.5 * (corr_frob_sqr - corr_diag_sqr)
            return loss 