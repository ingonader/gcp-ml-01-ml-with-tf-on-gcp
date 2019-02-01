
# coding: utf-8

# <h1> 2d. Distributed training and monitoring </h1>
# 
# In this notebook, we refactor to call ```train_and_evaluate``` instead of hand-coding our ML pipeline. This allows us to carry out evaluation as part of our training loop instead of as a separate step. It also adds in failure-handling that is necessary for distributed training capabilities.
# 
# We also use TensorBoard to monitor the training.

# In[1]:


import datalab.bigquery as bq
import tensorflow as tf
import numpy as np
import shutil
from google.datalab.ml import TensorBoard
print(tf.__version__)


# <h2> Input </h2>
# 
# Read data created in Lab1a, but this time make it more general, so that we are reading in batches.  Instead of using Pandas, we will use add a filename queue to the TensorFlow graph.

# In[6]:


CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

def read_dataset(filename, mode, batch_size = 512):
  def decode_csv(value_column):
    columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)
    return features, label

  # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
  filenames_dataset = tf.data.Dataset.list_files(filename)
  # Read lines from text files
  textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
  # Parse text lines as comma-separated values (CSV)
  dataset = textlines_dataset.map(decode_csv)

  # Note:
  # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
  # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

  if mode == tf.estimator.ModeKeys.TRAIN:
      num_epochs = None # indefinitely
      dataset = dataset.shuffle(buffer_size = 10 * batch_size)
  else:
      num_epochs = 1 # end-of-input after this

  dataset = dataset.repeat(num_epochs).batch(batch_size)

  return dataset


# <h2> Create features out of input data </h2>
# 
# For now, pass these through.  (same as previous lab)

# In[7]:


INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]

def add_more_features(feats):
  # Nothing to add (yet!)
  return feats

feature_cols = add_more_features(INPUT_COLUMNS)


# <h2> Serving input function </h2>
# Defines the expected shape of the JSON feed that the modelwill receive once deployed behind a REST API in production.

# In[8]:


# Defines the expected shape of the JSON feed that the model
# will receive once deployed behind a REST API in production.

## TODO: Create serving input function
def serving_input_fn():
    #ADD CODE HERE
    json_feature_placeholders = {
        'pickuplon'  : tf.placeholder(tf.float32, [None]),
        'pickuplat'  : tf.placeholder(tf.float32, [None]),
        'dropofflat' : tf.placeholder(tf.float32, [None]),
        'dropofflon' : tf.placeholder(tf.float32, [None]),
        'passengers' : tf.placeholder(tf.float32, [None])
    }
    
    ## transform data here, if needed (not needed here)
    features = json_feature_placeholders
    
    ## make features a 'ServingInputReceiver' and return it:
    ## (this will create a tensorflow node, which reads data
    ## consecutively when the graph is executed):
    return tf.estimator.export.ServingInputReceiver(features, json_feature_placeholders)


# <h2> tf.estimator.train_and_evaluate </h2>

# In[12]:


## TODO: Create train and evaluate function using tf.estimator
def train_and_evaluate(output_dir, num_train_steps):
    #ADD CODE HERE
    ## define estimator:
    estimator = tf.estimator.LinearRegressor(
        feature_columns = feature_cols,
        model_dir = output_dir
    )
    ## define the train spec, 
    ## which specifies the input function and max_steps
    ## (and possibly some hooks):
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = num_train_steps
    )
    
    ## define the exporter, which is needed for understanding
    ## json data coming in when model is deployed
    ## (serving time inputs); LatestExporter takes the latest
    ## checkpoint of the model:
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    
    ## define the eval spec (evaluation data input function):
    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL),
        steps = None, 
        start_delay_secs = 1, # start evaluating after n seconds
        throttle_secs = 10,   # evaluate every n seconds
        exporters = exporter   # using the model specified in the exporter (?)
    )
    
    ## finally, call the train_and_evaluate function in the tensorflow package:
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# <h2> Monitoring with TensorBoard </h2>
# <br/>
# Use "refresh" in Tensorboard during training to see progress.

# In[13]:


OUTDIR = 'taxi_trained'
TensorBoard().start(OUTDIR)


# <h2>Run training</h2>

# In[14]:


# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 2000)


# <h4> You can now shut Tensorboard down </h4>

# In[15]:


# to list Tensorboard instances
TensorBoard().list()


# In[17]:


# to stop TensorBoard fill the correct pid below
TensorBoard().stop(3864)
print("Stopped Tensorboard")


# ## Challenge Exercise
# 
# Modify your solution to the challenge exercise in c_dataset.ipynb appropriately.

# In[37]:


import datalab.bigquery as bq
import tensorflow as tf
import numpy as np
import shutil
print(tf.__version__)

from google.datalab.ml import TensorBoard

import math
import pandas as pd

def gen_data(n):
  r = np.random.uniform(.5, 2, n)
  h = np.random.uniform(.5, 2, n)
  v = r ** 2 * h * math.pi
  dat = pd.DataFrame({
    'r': np.round(r, 1),
    'h': np.round(h, 1),
    'v': np.round(v, 1)
  })
  return dat


dat = gen_data(5000)
dat_eval = gen_data(1000)
dat_test = gen_data(1000)

dat.head(n = 2)


# In[38]:


## write data to file:
dat.to_csv('dat-cyl-train.csv', header = False, index = False)
dat_eval.to_csv('dat-cyl-eval.csv', header = False, index = False)
dat_test.to_csv('dat-cyl-test.csv', header = False, index = False)


# In[39]:


get_ipython().system('ls -l dat-cyl*.csv')


# In[40]:


get_ipython().system('head -n 5 dat-cyl-train.csv')


# In[41]:


CSV_COLUMNS = ['h', 'r', 'v']
DEFAULTS = [[0.0], [0.0], [0.0]]
LABEL_COLUMN = 'v'

## create a function to read the dataset from disk:
def read_dataset(filename, mode, batch_size = 512):
  ## Add CSV decoder function: 
  def decode_csv(row):
    columns = tf.decode_csv(row, record_defaults = DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)
    return features, label
  
  ## create dataset:
  ## read list of file names:
  filenames_dataset = tf.data.Dataset.list_files(filename, shuffle = False)
  ## read lines from this dataset list:
  textlines_dataset = filenames_dataset.flat_map(
    tf.data.TextLineDataset   ## function that returns individual text lines of the file
  )
  ## then, decode each line:
  dataset = textlines_dataset.map(decode_csv)

  # Note:
  # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
  # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

  if mode == tf.estimator.ModeKeys.TRAIN:
    num_epochs = None ## loop indefinitely
    dataset = dataset.shuffle(buffer_size = 10 * batch_size, seed = 2) ## and shuffle for training
  else:
    num_epochs = 1 ## only loop once through dataset
    
  ## repeat data set as needed (training) and form batches of data:
  dataset = dataset.repeat(num_epochs).batch(batch_size)
  return dataset

# ## define function that gets training input, using the read_dataset function above:
# ## (not done here; will be done with lambda function later in train_spec and eval_spec)
# 
# def get_train_input_fn():
#   return read_dataset('./dat-cyl-train.csv', mode = tf.estimator.ModeKeys.TRAIN)
# 
# def get_valid_input_fn():
#   return read_dataset('./dat-cyl-eval.csv', mode = tf.estimator.ModeKeys.EVAL)


# In[42]:


## create input features:
INPUT_COLUMNS = [
  tf.feature_column.numeric_column('h'),
  tf.feature_column.numeric_column('r'),
]

def add_more_features(feats):
  # Nothing to add (yet!)
  return feats

## currently, don't apply any feature transformations:
feature_cols = add_more_features(INPUT_COLUMNS)


# In[43]:


## create serving input function:

def serving_input_fn():
  json_feature_placeholders = {
    'h' : tf.placeholder(tf.float32, [None]),
    'r' : tf.placeholder(tf.float32, [None])
  }
  
  ## transform features here: (not needed for now)
  features = json_feature_placeholders
  
  ## make features a 'ServingInputReceiver' and return it:
  ## (this will create a tensorflow node, which reads data
  ## consecutively when the graph is executed):
  return tf.estimator.export.ServingInputReceiver(features, json_feature_placeholders)


# In[44]:


## define train_and_evaluate function 
## for monitoring job with tensorboard:
def get_estimator(output_dir):
  estimator = tf.estimator.DNNRegressor(
    feature_columns = feature_cols, 
    hidden_units = [4, 8, 4], 
    model_dir = output_dir)
  return estimator

def train_and_evaluate(output_dir, num_train_steps):
  ## define estimator:
  estimator = get_estimator(output_dir)

  ## define the train spec, 
  ## which specifies the input function and max_steps
  ## (and possibly some hooks):
  train_spec = tf.estimator.TrainSpec(
    input_fn = lambda: read_dataset('./dat-cyl-train.csv', 
                                    mode = tf.estimator.ModeKeys.TRAIN),
    max_steps = num_train_steps)
  
  ## define the exporter, which is needed for understanding
  ## json data coming in when model is deployed
  ## (serving time inputs); LatestExporter takes the latest
  ## checkpoint of the model:
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

  ## define the eval spec (evaluation data input function):
  eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: read_dataset('./dat-cyl-eval.csv', 
                                    mode = tf.estimator.ModeKeys.EVAL),
    steps = None, 
    start_delay_secs = 1,   # start evaluating after n seconds
    throttle_secs = 10,     # evaluate every n seconds
    exporters = exporter)   # using the model specified in the exporter (?)
  
  ## finally, call the train_and_evaluate function in the tensorflow package:
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  


# In[45]:


## monitor with tensorboard:
#from google.datalab.ml import TensorBoard

OUTDIR = './cyl_trained'
TensorBoard().start(OUTDIR)


# In[46]:


# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 2000)


# In[47]:


## load model from disk:
model = get_estimator(OUTDIR)


# In[48]:


## RMSE:
metrics = model.evaluate(input_fn = lambda: read_dataset('./dat-cyl-eval.csv', 
                                    mode = tf.estimator.ModeKeys.EVAL), 
                         steps = None)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))


# In[49]:


## make prediction iterator:
pred_iter = model.predict(input_fn = lambda: read_dataset('./dat-cyl-eval.csv', 
                                    mode = tf.estimator.ModeKeys.EVAL))
dat_pred = pd.DataFrame(columns = ['v_true', 'v_pred'])

## [[?]]
## how to get correct true labels in distributed training?
## maybe use different input_fn for predict, starting from a 
## pandas df for easier data inspection?

## predict a few values to get correlation:
for i in range(1000):
  dat_pred = dat_pred.append({
    'v_true' : dat_eval['v'][i],
    'v_pred' : next(pred_iter)['predictions'][0]
  }, ignore_index = True)
  #print(dat_eval['v'][i], next(pred_iter)['predictions'][0])
  
dat_pred.head(n = 5)


# In[50]:


dat_pred.corr()


# In[51]:


# to list Tensorboard instances
TensorBoard().list()


# In[52]:


pids_df = TensorBoard.list()
if not pids_df.empty:
    for pid in pids_df['pid']:
        TensorBoard().stop(pid)
        print('Stopped TensorBoard with pid {}'.format(pid))


# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License