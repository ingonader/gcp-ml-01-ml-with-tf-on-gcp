
# coding: utf-8

# <h1> 2c. Loading large datasets progressively with the tf.data.Dataset </h1>
# 
# In this notebook, we continue reading the same small dataset, but refactor our ML pipeline in two small, but significant, ways:
# <ol>
# <li> Refactor the input to read data from disk progressively.
# <li> Refactor the feature creation so that it is not one-to-one with inputs.
# </ol>
# <br/>
# The Pandas function in the previous notebook first read the whole data into memory -- on a large dataset, this won't be an option.

# In[25]:


import datalab.bigquery as bq
import tensorflow as tf
import numpy as np
import shutil
print(tf.__version__)


# <h2> 1. Refactor the input </h2>
# 
# Read data created in Lab1a, but this time make it more general, so that we can later handle large datasets. We use the Dataset API for this. It ensures that, as data gets delivered to the model in mini-batches, it is loaded from disk only when needed.

# In[26]:


CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# TODO: Create an appropriate input function read_dataset
def read_dataset(filename, mode, batch_size = 512):
    #TODO Add CSV decoder function and dataset creation and methods
    def decode_csv(row):
        columns = tf.decode_csv(row, record_defaults = DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop('fare_amount')
        return features, label

    ## read list of file names:
    filenames_dataset = tf.data.Dataset.list_files(filename, shuffle = False)
    ## read lines from this dataset:
    textlines_dataset = filenames_dataset.flat_map(
        tf.data.TextLineDataset
    )
    ## decode each line:
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

     
def get_train_input_fn():
    return read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN)

def get_valid_input_fn():
    return read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL)


# <h2> 2. Refactor the way features are created. </h2>
# 
# For now, pass these through (same as previous lab).  However, refactoring this way will enable us to break the one-to-one relationship between inputs and features.

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


# <h2> Create and train the model </h2>
# 
# Note that we train for num_steps * batch_size examples.

# In[8]:


tf.logging.set_verbosity(tf.logging.INFO)
OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.LinearRegressor(
      feature_columns = feature_cols, model_dir = OUTDIR)
model.train(input_fn = get_train_input_fn, steps = 200)


# <h3> Evaluate model </h3>
# 
# As before, evaluate on the validation data.  We'll do the third refactoring (to move the evaluation into the training loop) in the next lab.

# In[9]:


metrics = model.evaluate(input_fn = get_valid_input_fn, steps = None)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))


# ## Challenge Exercise
# 
# Create a neural network that is capable of finding the volume of a cylinder given the radius of its base (r) and its height (h). Assume that the radius and height of the cylinder are both in the range 0.5 to 2.0. Unlike in the challenge exercise for b_estimator.ipynb, assume that your measurements of r, h and V are all rounded off to the nearest 0.1. Simulate the necessary training dataset. This time, you will need a lot more data to get a good predictor.
# 
# Hint (highlight to see):
# <p style='color:white'>
# Create random values for r and h and compute V. Then, round off r, h and V (i.e., the volume is computed from the true value of r and h; it's only your measurement that is rounded off). Your dataset will consist of the round values of r, h and V. Do this for both the training and evaluation datasets.
# </p>
# 
# Now modify the "noise" so that instead of just rounding off the value, there is up to a 10% error (uniformly distributed) in the measurement followed by rounding off.

# In[20]:


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


# In[21]:


FEATURES = ['h', 'r']
LABEL = 'v'

featcols = [
  tf.feature_column.numeric_column("h"),
  tf.feature_column.numeric_column("r")
]

def make_train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True
  )

def make_eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = False
  )

def make_prediction_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    batch_size = 128,
    shuffle = False
  )

tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(tf.logging.WARN)

OUTDIR = 'cyl_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

# TODO: Train a linear regression model
#model = tf.estimator.LinearRegressor(featcols, OUTDIR) #ADD CODE HERE
model = tf.estimator.DNNRegressor(feature_columns = featcols, hidden_units = [4, 8, 4], model_dir = OUTDIR) #ADD CODE HERE

model.train(  #ADD CODE HERE
  make_train_input_fn(dat, num_epochs = 10), 
  max_steps = 100000
)


# In[22]:


## RMSE:
def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  #print('RMSE on dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, dat_eval)


# In[23]:


## predict some values and get correlation:
pred_iter = model.predict(make_prediction_input_fn(dat_eval))
dat_pred = pd.DataFrame(columns = ['v_true', 'v_pred'])
for i in range(20):
  dat_pred = dat_pred.append({
    'v_true' : dat_eval['v'][i],
    'v_pred' : next(pred_iter)['predictions'][0]
  }, ignore_index = True)
  #print(dat_eval['v'][i], next(pred_iter)['predictions'][0])
  
dat_pred.head(n = 5)


# In[24]:


dat_pred.corr()


# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License