
# coding: utf-8

# <h1>2b. Machine Learning using tf.estimator </h1>
# 
# In this notebook, we will create a machine learning model using tf.estimator and evaluate its performance.  The dataset is rather small (7700 samples), so we can do it all in-memory.  We will also simply pass the raw data in as-is. 

# In[5]:


import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

print(tf.__version__)


# Read data created in the previous chapter.

# In[2]:


# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv('./taxi-train.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('./taxi-valid.csv', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv('./taxi-test.csv', header = None, names = CSV_COLUMNS)


# In[3]:


df_train.head(n = 2)
#type(df_train[LABEL])  # pandas.core.series.Series


# <h2> Train and eval input functions to read from Pandas Dataframe </h2>

# In[4]:


# TODO: Create an appropriate input_fn to read the training data
def make_train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True
  )


# In[5]:


# TODO: Create an appropriate input_fn to read the validation data
def make_eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = False
  )


# Our input function for predictions is the same except we don't provide a label

# In[6]:


# TODO: Create an appropriate prediction_input_fn
def make_prediction_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    batch_size = 128,
    shuffle = False
  )


# ### Create feature columns for estimator

# In[7]:


df_train.head(n = 2)


# In[8]:


# TODO: Create feature columns
featcols = [
  tf.feature_column.numeric_column("pickuplon"),
  tf.feature_column.numeric_column("pickuplat"),
  tf.feature_column.numeric_column("dropofflon"),
  tf.feature_column.numeric_column("dropofflat"),
  tf.feature_column.numeric_column("passengers")
  #tf.feature_column.numeric_column("key")
]


# <h3> Linear Regression with tf.Estimator framework </h3>

# In[9]:


tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

# TODO: Train a linear regression model
model = tf.estimator.LinearRegressor(featcols, OUTDIR) #ADD CODE HERE

model.train(  #ADD CODE HERE
  make_train_input_fn(df_train, num_epochs = 10),
  max_steps = 100000
)


# In[10]:


## start tensorboard:
from google.datalab.ml import TensorBoard as tb

tb.start(OUTDIR)


# Evaluate on the validation data (we should defer using the test data to after we have selected a final model).

# In[11]:


def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  #print('RMSE on dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, df_valid)


# This is nowhere near our benchmark (RMSE of $6 or so on this data), but it serves to demonstrate what TensorFlow code looks like.  Let's use this model for prediction.

# In[12]:


# TODO: Predict from the estimator model we trained using test dataset
pred_iter = model.predict(make_prediction_input_fn(df_test))
for i in range(20):
  print(next(pred_iter))


# This explains why the RMSE was so high -- the model essentially predicts the same amount for every trip.  Would a more complex model help? Let's try using a deep neural network.  The code to do this is quite straightforward as well.

# <h3> Deep Neural Network regression </h3>

# In[13]:


# TODO: Copy your LinearRegressor estimator and replace with DNNRegressor. 
# Remember to add a list of hidden units i.e. [32, 8, 2]

tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

# TODO: Train a linear regression model
model = tf.estimator.DNNRegressor(feature_columns = featcols, hidden_units = [32, 8, 2], model_dir = OUTDIR) #ADD CODE HERE

model.train(  #ADD CODE HERE
  make_train_input_fn(df_train, num_epochs = 30),
  max_steps = 100000
)


# In[14]:


def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  #print('RMSE on dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, df_valid)


# In[15]:


pred_iter = model.predict(make_prediction_input_fn(df_test))
for i in range(20):
  print(next(pred_iter))


# We are not beating our benchmark with either model ... what's up?  Well, we may be using TensorFlow for Machine Learning, but we are not yet using it well.  That's what the rest of this course is about!
# 
# But, for the record, let's say we had to choose between the two models. We'd choose the one with the lower validation error. Finally, we'd measure the RMSE on the test data with this chosen model.

# <h2> Benchmark dataset </h2>
# 
# Let's do this on the benchmark dataset.

# In[16]:


import google.datalab.bigquery as bq
import numpy as np
import pandas as pd

def create_query(phase, EVERY_N):
  """
  phase: 1 = train 2 = valid
  """
  base_query = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  EXTRACT(DAYOFWEEK FROM pickup_datetime) * 1.0 AS dayofweek,
  EXTRACT(HOUR FROM pickup_datetime) * 1.0 AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count * 1.0 AS passengers,
  CONCAT(CAST(pickup_datetime AS STRING), CAST(pickup_longitude AS STRING), CAST(pickup_latitude AS STRING), CAST(dropoff_latitude AS STRING), CAST(dropoff_longitude AS STRING)) AS key
FROM
  `nyc-tlc.yellow.trips`
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  """

  if EVERY_N == None:
    if phase < 2:
      # Training
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), 4) < 2".format(base_query)
    else:
      # Validation
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), 4) = {1}".format(base_query, phase)
  else:
    query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), {1}) = {2}".format(base_query, EVERY_N, phase)
    
  return query

query = create_query(2, 100000)
df = bq.Query(query).execute().result().to_dataframe()
df.head(n = 2)


# In[17]:


print_rmse(model, df)


# RMSE on benchmark dataset is <b>9.41</b> (your results will vary because of random seeds).
# 
# This is not only way more than our original benchmark of 6.00, but it doesn't even beat our distance-based rule's RMSE of 8.02.
# 
# Fear not -- you have learned how to write a TensorFlow model, but not to do all the things that you will have to do to your ML model performant. We will do this in the next chapters. In this chapter though, we will get our TensorFlow model ready for these improvements.
# 
# In a software sense, the rest of the labs in this chapter will be about refactoring the code so that we can improve it.

# ## Challenge Exercise
# 
# Create a neural network that is capable of finding the volume of a cylinder given the radius of its base (r) and its height (h). Assume that the radius and height of the cylinder are both in the range 0.5 to 2.0. Simulate the necessary training dataset.
# <p>
# Hint (highlight to see):
# <p style='color:white'>
# The input features will be r and h and the label will be $\pi r^2 h$
# Create random values for r and h and compute V.
# Your dataset will consist of r, h and V.
# Then, use a DNN regressor.
# Make sure to generate enough data.
# </p>

# In[11]:


import math
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

def gen_data(n):
  r = np.random.uniform(.5, 2, n)
  h = np.random.uniform(.5, 2, n)
  v = r ** 2 * h * math.pi
  dat = pd.DataFrame({
    'r': r,
    'h': h,
    'v': v
  })
  return dat


dat = gen_data(5000)
dat_eval = gen_data(1000)
dat_test = gen_data(1000)

dat.head(n = 2)


# In[12]:


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


# In[13]:


## RMSE:
def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  #print('RMSE on dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, dat_eval)


# In[14]:


## predict some values and get correlation:
pred_iter = model.predict(make_prediction_input_fn(dat_eval))
dat_pred = pd.DataFrame(columns = ['v_true', 'v_pred'])
for i in range(20):
  dat_pred = dat_pred.append({
    'v_true' : dat_eval['v'][i],
    'v_pred' : next(pred_iter)['predictions'][0]
  }, ignore_index = True)
  #print(dat_eval['v'][i], next(pred_iter)['predictions'][0])
  
dat_pred.head(n = 2)


# In[15]:


dat_pred.corr()


# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License