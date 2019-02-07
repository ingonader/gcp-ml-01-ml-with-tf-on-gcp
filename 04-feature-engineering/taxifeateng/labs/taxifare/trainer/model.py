#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = 'fare_amount,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,key'.split(',')
LABEL_COLUMN = 'fare_amount'
KEY_FEATURE_COLUMN = 'key'
DEFAULTS = [[0.0], ['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # TODO: Define feature columns for dayofweek, hourofday, pickuplon, pickuplat, dropofflat, dropofflon, passengers

    ## categorical columns:
    tf.feature_column.categorical_column_with_vocabulary_list(
        'dayofweek', 
        vocabulary_list = ['Sun', 'Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat']),
    tf.feature_column.categorical_column_with_identity('hourofday', num_buckets = 24),
    
    ## numeric columns:
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
    
    # TODO: Add any engineered columns here
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]

# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     TODO: Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model, i.e. [wide_cols] & [deep_cols].
    """
    
    ## [[here]] -- change "euclidean" to "pcount" (12:49)

    ## retrieve input features into separate variables:
    #(dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean) = INPUT_COLUMNS
    (dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean) = INPUT_COLUMNS
    
    ## transform features: 
    ## bucketize lats & lons:
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)
    
    ## feature crosses (a.k.a. interactions):
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets ** 2)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets ** 2)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)
    day_hr = tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)
    
    ## define linear features (a.k.a. wide columns):
    linear_feature_columns = [
        ## feature crosses:
        dloc, ploc, pd_pair, day_hr,
        ## sparse columns:
        dayofweek, hourofday,
        ## anything with a linear relationship:
        pcount
    ]
    ## define dnn features (a.k.a. deep columns):
    dnn_feature_columns = [
        ## embedding column to "group" similar columns of feature crosses:
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),
        ## numeric columns:
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]
    
    ## define run config for a much longer interval
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 30,
                                        keep_checkpoint_max = 3)
    ## define DNN wide & deep estimator:
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        linear_feature_columns = linear_feature_columns, ## wide features
        dnn_feature_columns = dnn_feature_columns,       ## deep features
        dnn_hidden_units = hidden_units,
        config = run_config
    )
    # add extra evaluation metric for hyperparameter tuning
    #estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator # TODO: Add estimator definition here



# Create feature engineering function that will be used in the input and serving input functions
def add_engineered(features):
    # TODO: Add any engineered features to the dict
    lat1 = features['pickuplat']
    lat2 = features['dropofflat']
    lon1 = features['pickuplon']
    lon2 = features['dropofflon']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    
    ## add features to feature vector
    ## for distance with sign that indicates direction:
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
    features['euclidean'] = dist
    
    return features   



# # Create serving input function to be able to serve predictions
# def serving_input_fn():
#     # ## code in prepared file for lab:
#     # feature_placeholders = {  
#     #     # TODO: What features will user provide? What will their types be?
#     #     ## numeric features:
#     #     column.name : tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:7]
#     # }
#     ## code trials:
#     feature_placeholders = {}
#     ## non-numeric features:
#     feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
#     feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])
#     ## numeric features:
#     feature_placeholders['pickuplat'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['pickuplon'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['dropofflat'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['dropofflon'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['passengers'] = tf.placeholder(tf.float32, [None])
# 
#     # TODO: Add any extra placeholders for inputs you'll generate
#     features = add_engineered(feature_placeholders.copy())
# 
#     # ## [[?]] this part is not part of the model solution... 
#     # what is going on? --> hence, commented out.
#     # features = {
#     #     key: tf.expand_dims(tensor, -1)
#     #     for key, tensor in feature_placeholders.items()
#     # }
#     return tf.estimator.export.ServingInputReceiver(
#       features, # TODO: Wrap this with a call to add_engineered
#       feature_placeholders
#     )

## new implementation -- [[here]]:
## similar to course video:

# Create serving input function to be able to serve predictions
def serving_input_fn():
    feature_placeholders = {  
        ## numeric features:
        ## (ignoring the first two columns):
        column.name : tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:]
    }
    feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
    feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])
    ## no add engineered here, but below
    ## needs to be done _after_ the features are created!!!

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
      add_engineered(features), 
      feature_placeholders
    )

# Create input function to load data into datasets
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label # TODO: Wrap this with a call to add_engineered
        
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn



## Create estimator train and evaluate function
#def train_and_evaluate(args):
#    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'])
#    train_spec = tf.estimator.TrainSpec(
#        input_fn = read_dataset(
#            filename = args['train_data_paths'],
#            mode = tf.estimator.ModeKeys.TRAIN,
#            batch_size = args['train_batch_size']),
#        max_steps = args['train_steps'])
#    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
#    eval_spec = tf.estimator.EvalSpec(
#        input_fn = read_dataset(
#            filename = args['eval_data_paths'],
#            mode = tf.estimator.ModeKeys.EVAL,
#            batch_size = args['eval_batch_size']),
#        steps = None,
#        exporters = exporter)
#    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

## new implementation copied from coursera course video (18:45):
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'].split(' '))
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = None,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def get_eval_metrics():
    return {
        'rmse': tflearn.MetricSpec(metric_fn = metrics.streaming_root_mean_squared_error),
        'training/hptuning/metric': tflearn.MetricSpec(metric_fn = metrics.streaming_root_mean_squared_error),
    }




