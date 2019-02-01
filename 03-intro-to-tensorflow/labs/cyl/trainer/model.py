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
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

# List the CSV columns
CSV_COLUMNS = ['h', 'r', 'v']

#Choose which column is your label
LABEL_COLUMN = 'v'

# Set the default values for each CSV column in case there is a missing value
DEFAULTS = [[0.0], [0.0], [0.0]]

# Create an input function that stores your data into a dataset
# TODO: Add input function
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(row):
            columns = tf.decode_csv(row, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
        
        ## Create a list of files that match pattern:
        ## (not flat_map needed, seemingly):
        file_list = tf.gfile.Glob(filename)
        
        ## Create a dataset from that file list:
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        
        ## return an iterator, that will return a batch of data:
        ## (only run once, will create a tf node that does this):
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn
        


# Define your feature columns
INPUT_COLUMNS = [
  tf.feature_column.numeric_column('h'),
  tf.feature_column.numeric_column('r'),
]

# Create a function that will augment your feature set
def add_more_features(feats):
    # Nothing to add (yet!)
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)

# Create your serving input function so that your trained model will be able to serve predictions
def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    ## transform data here, if needed (not needed here):
    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# Create an estimator that we are going to train and evaluate
# TODO: Create tf.estimator.DNNRegressor train and evaluate function passing args['parsed_argument'] from task.py
def train_and_evaluate(args):
    ## define estimator:
    estimator = tf.estimator.DNNRegressor(
        feature_columns = feature_cols,
        hidden_units = args['hidden_units'],
        model_dir = args['output_dir']
    )
    ## define the train spec, 
    ## which specifies the input function and max_steps
    ## (and possibly some hooks):
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(args['train_data_paths'],  ## read_dataset will now return a function!
                                batch_size = args['train_batch_size'], 
                                mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps']
    )
    ## define the exporter, which is needed for understanding
    ## json data coming in when model is deployed
    ## (serving time inputs); LatestExporter takes the latest
    ## checkpoint of the model:
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    ## define the eval spec (evaluation data input function):
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(args['eval_data_paths'], 
                                mode = tf.estimator.ModeKeys.EVAL),
        steps = None, 
        start_delay_secs = args['eval_delay_secs'],  # start evaluating after n seconds
        throttle_secs =    args['throttle_secs'],    # evaluate every n seconds
        exporters = exporter   # using the model specified in the exporter (?)
    )
    
    ## finally, call the train_and_evaluate function in the tensorflow package:
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    




