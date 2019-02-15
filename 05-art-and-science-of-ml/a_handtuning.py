
# coding: utf-8

# # Hand tuning hyperparameters

# **Learning Objectives:**
#   * Use the `LinearRegressor` class in TensorFlow to predict median housing price, at the granularity of city blocks, based on one input feature
#   * Evaluate the accuracy of a model's predictions using Root Mean Squared Error (RMSE)
#   * Improve the accuracy of a model by hand-tuning its hyperparameters

# The data is based on 1990 census data from California. This data is at the city block level, so these features reflect the total number of rooms in that block, or the total number of people who live on that block, respectively.  Using only one input feature -- the number of rooms -- predict house value.

# ## Set Up
# In this first cell, we'll load the necessary libraries.

# In[9]:


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

# In[10]:


df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")


# ## Examine the data
# 
# It's a good idea to get to know your data a little bit before you work with it.
# 
# We'll print out a quick summary of a few useful statistics on each column.
# 
# This will include things like mean, standard deviation, max, min, and various quantiles.

# In[11]:


df.head()


# In[12]:


df.describe()


# In this exercise, we'll be trying to predict median_house_value. It will be our label (sometimes also called a target). Can we use total_rooms as our input feature?  What's going on with the values for that feature?
# 
# This data is at the city block level, so these features reflect the total number of rooms in that block, or the total number of people who live on that block, respectively.  Let's create a different, more appropriate feature.  Because we are predicing the price of a single house, we should try to make all our features correspond to a single house as well

# In[13]:


df['num_rooms'] = df['total_rooms'] / df['households']
df.describe()


# In[14]:


# Split into train and eval
np.random.seed(seed=1) #makes split reproducible
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]


# ## Build the first model
# 
# In this exercise, we'll be trying to predict `median_house_value`. It will be our label (sometimes also called a target). We'll use `num_rooms` as our input feature.
# 
# To train our model, we'll use the [LinearRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor) estimator. The Estimator takes care of a lot of the plumbing, and exposes a convenient way to interact with data, training, and evaluation.

# In[15]:


OUTDIR = './housing_trained'
def train_and_evaluate(output_dir, num_train_steps):
  estimator = tf.estimator.LinearRegressor(
                       model_dir = output_dir, 
                       feature_columns = [tf.feature_column.numeric_column('num_rooms')])
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}
  estimator = tf.contrib.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = tf.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                              y = traindf["median_house_value"],  # note the scaling
                                              num_epochs = None,
                                              shuffle = True),
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = tf.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                              y = evaldf["median_house_value"],  # note the scaling
                                              num_epochs = 1,
                                              shuffle = False),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  
# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100)


# ## 1. Scale the output
# Let's scale the target values so that the default parameters are more appropriate.

# In[16]:


SCALE = 100000
OUTDIR = './housing_trained'
def train_and_evaluate(output_dir, num_train_steps):
  estimator = tf.estimator.LinearRegressor(
                       model_dir = output_dir, 
                       feature_columns = [tf.feature_column.numeric_column('num_rooms')])
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
  estimator = tf.contrib.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = tf.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                              y = traindf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = None,
                                              shuffle = True),
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = tf.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                              y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = 1,
                                              shuffle = False),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100)


# ## 2. Change learning rate and batch size
# Can you come up with better parameters? 

# In[17]:


SCALE = 100000
OUTDIR = './housing_trained'
def train_and_evaluate(output_dir, num_train_steps):
  myopt = tf.train.FtrlOptimizer(learning_rate = 0.2) # note the learning rate
  estimator = tf.estimator.LinearRegressor(
                       model_dir = output_dir, 
                       feature_columns = [tf.feature_column.numeric_column('num_rooms')],
                       optimizer = myopt)
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
  estimator = tf.contrib.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = tf.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                              y = traindf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = None,
                                              batch_size = 512, # note the batch size
                                              shuffle = True),
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = tf.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                              y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = 1,
                                              shuffle = False),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100) 


# ### Is there a standard method for tuning the model?
# 
# This is a commonly asked question. The short answer is that the effects of different hyperparameters is data dependent.  So there are no hard and fast rules; you'll need to run tests on your data.
# 
# Here are a few rules of thumb that may help guide you:
# 
#  * Training error should steadily decrease, steeply at first, and should eventually plateau as training converges.
#  * If the training has not converged, try running it for longer.
#  * If the training error decreases too slowly, increasing the learning rate may help it decrease faster.
#    * But sometimes the exact opposite may happen if the learning rate is too high.
#  * If the training error varies wildly, try decreasing the learning rate.
#    * Lower learning rate plus larger number of steps or larger batch size is often a good combination.
#  * Very small batch sizes can also cause instability.  First try larger values like 100 or 1000, and decrease until you see degradation.
# 
# Again, never go strictly by these rules of thumb, because the effects are data dependent.  Always experiment and verify.

# ### 3: Try adding more features
# 
# See if you can do any better by adding more features.
# 
# Don't take more than 5 minutes on this portion.