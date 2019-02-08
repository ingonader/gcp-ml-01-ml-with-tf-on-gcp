
# coding: utf-8

# # Trying out features

# **Learning Objectives:**
#   * Improve the accuracy of a model by adding new features with the appropriate representation

# The data is based on 1990 census data from California. This data is at the city block level, so these features reflect the total number of rooms in that block, or the total number of people who live on that block, respectively.

# ## Set Up
# In this first cell, we'll load the necessary libraries.

# In[1]:


import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# Next, we'll load our data set.

# In[2]:


df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")


# ## Examine and split the data
# 
# It's a good idea to get to know your data a little bit before you work with it.
# 
# We'll print out a quick summary of a few useful statistics on each column.
# 
# This will include things like mean, standard deviation, max, min, and various quantiles.

# In[3]:


df.head()


# In[4]:


df.describe()


# Now, split the data into two parts -- training and evaluation.

# In[5]:


np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]


# ## Training and Evaluation
# 
# In this exercise, we'll be trying to predict **median_house_value** It will be our label (sometimes also called a target).
# 
# We'll modify the feature_cols and input function to represent the features you want to use.
# 
# Hint: Some of the features in the dataframe aren't directly correlated with median_house_value (e.g. total_rooms) but can you think of a column to divide it by that we would expect to be correlated with median_house_value?

# In[6]:


def add_more_features(df):
  df['rooms_per_hh'] = df['total_rooms'] / df['households']
  df['bedrooms_per_hh'] = df['total_bedrooms'] / df['households']
  df['pop_per_hh'] = df['population'] / df['households']
  # TODO: Add more features to the dataframe
  return df

add_more_features(df).head()


# In[37]:


# Create pandas input function
def make_input_fn(df, num_epochs, shuffle = True):
  return tf.estimator.inputs.pandas_input_fn(
    x = add_more_features(df),
    y = df['median_house_value'] / 100000, # will talk about why later in the course
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = shuffle,
    queue_capacity = 1000,
    num_threads = 1
  )


# In[59]:


# Define your feature columns
FEATURE_COLS_NUM = ['housing_median_age', 'median_income', 'rooms_per_hh', 'bedrooms_per_hh', 'pop_per_hh']
FEATURE_COLS_PP = ['latitude']  ## need some kind of preprocessing
FEATURE_COLS = FEATURE_COLS_NUM + FEATURE_COLS_PP
print(FEATURE_COLS)

def create_feature_cols():
  return [
    # TODO: Define additional feature columns
    # Hint: Are there any features that would benefit from bucketizing?
    tf.feature_column.numeric_column(i) for i in FEATURE_COLS_NUM ] + \
  [
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'),
                                        boundaries = np.arange(32.0, 42.0, 1).
                                        tolist())
  ]

print(create_feature_cols())


# In[60]:


def get_estimator(output_dir):
  estimator = tf.estimator.LinearRegressor(
    feature_columns = create_feature_cols(), 
    #hidden_units = [4, 8, 4], 
    model_dir = output_dir)
  return estimator

# Create estimator train and evaluate function
def train_and_evaluate(output_dir, num_train_steps):
  # TODO: Create tf.estimator.LinearRegressor, train_spec, eval_spec, and train_and_evaluate using your feature columns
  ## define estimator:
  estimator = get_estimator(output_dir)
  
  ## define the train spec, 
  ## which specifies the input function and max_steps
  ## (and possibly some hooks):
  train_spec = tf.estimator.TrainSpec(
    input_fn = make_input_fn(df = traindf, num_epochs = None), ## [[?]]
    max_steps = num_train_steps
  )
  
  ## define the exporter, which is needed for understanding
  ## json data coming in when model is deployed
  ## (serving time inputs); LatestExporter takes the latest
  ## checkpoint of the model:
  #exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
   
  ## define the eval spec (evaluation data input function):
  eval_spec = tf.estimator.EvalSpec(
    input_fn = make_input_fn(df = evaldf, num_epochs = 1),
    steps = None,
    start_delay_secs = 1,
    throttle_secs = 10
  )
  
  ## call train_and_evaluate!
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# In[55]:


## try without train_and_evaluate first (and w/o tensorboard):
OUTDIR = './trained_model'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.LinearRegressor(create_feature_cols(), OUTDIR) #ADD CODE HERE

#model.train(
#  make_input_fn(traindf, num_epochs = 10),
#  max_steps = 100000
#)


# In[51]:


# Launch tensorboard
from google.datalab.ml import TensorBoard

OUTDIR = './trained_model'
TensorBoard().start(OUTDIR)


# In[61]:


# Run the model
shutil.rmtree(OUTDIR, ignore_errors = True)
train_and_evaluate(OUTDIR, 2000)


# In[62]:


pids_df = TensorBoard.list()
pids_df


# In[63]:


pids_df = TensorBoard.list()
if not pids_df.empty:
    for pid in pids_df['pid']:
        TensorBoard().stop(pid)
        print('Stopped TensorBoard with pid {}'.format(pid))


# In[64]:


## load model from disk:
model = get_estimator(OUTDIR)


# In[48]:


## RMSE:
metrics = model.evaluate(input_fn = make_input_fn(df = evaldf, num_epochs = 1, shuffle = False), 
                         steps = None)
print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))


# In[65]:


## make prediction iterator:
pred_iter = model.predict(input_fn = make_input_fn(df = evaldf, num_epochs = 1, shuffle = False))
dat_pred = pd.DataFrame(columns = ['v_true', 'v_pred'])

## [[?]]
## how to get correct true labels in distributed training?
## maybe use different input_fn for predict, starting from a 
## pandas df for easier data inspection?

## predict a few values to get correlation:
for i in range(1000):
  dat_pred = dat_pred.append({
    'v_true' : evaldf['median_house_value'].iloc[i],
    'v_pred' : next(pred_iter)['predictions'][0]
  }, ignore_index = True)
  #print(dat_eval['v'][i], next(pred_iter)['predictions'][0])
  
print(dat_pred.head(n = 5))
dat_pred.corr()
