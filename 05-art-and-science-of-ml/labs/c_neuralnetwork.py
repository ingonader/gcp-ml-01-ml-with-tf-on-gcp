
# coding: utf-8

# # Neural Network

# **Learning Objectives:**
#   * Use the `DNNRegressor` class in TensorFlow to predict median housing price

# The data is based on 1990 census data from California. This data is at the city block level, so these features reflect the total number of rooms in that block, or the total number of people who live on that block, respectively.
# <p>
# Let's use a set of features to predict house value.

# ## Set Up
# In this first cell, we'll load the necessary libraries.

# In[1]:


import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# Next, we'll load our data set.

# In[2]:


df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")


# ## Examine the data
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


# This data is at the city block level, so these features reflect the total number of rooms in that block, or the total number of people who live on that block, respectively.  Let's create a different, more appropriate feature.  Because we are predicing the price of a single house, we should try to make all our features correspond to a single house as well

# In[5]:


df['num_rooms'] = df['total_rooms'] / df['households']
df['num_bedrooms'] = df['total_bedrooms'] / df['households']
df['persons_per_house'] = df['population'] / df['households']
df.describe()


# In[6]:


df.drop(['total_rooms', 'total_bedrooms', 'population', 'households'], axis = 1, inplace = True)
df.describe()


# ## Build a neural network model
# 
# In this exercise, we'll be trying to predict `median_house_value`. It will be our label (sometimes also called a target). We'll use the remaining columns as our input features.
# 
# To train our model, we'll first use the [LinearRegressor](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearRegressor) interface. Then, we'll change to DNNRegressor
# 

# In[7]:


featcols = {
  colname : tf.feature_column.numeric_column(colname) \
    for colname in 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house'.split(',')
}
# Bucketize lat, lon so it's not so high-res; California is mostly N-S, so more lats than lons
featcols['longitude'] = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'),
                                                   np.linspace(-124.3, -114.3, 5).tolist())
featcols['latitude'] = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'),
                                                  np.linspace(32.5, 42, 10).tolist())


# In[8]:


featcols.keys()


# In[9]:


# Split into train and eval
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]

SCALE = 100000
BATCH_SIZE= 100
OUTDIR = './housing_trained'
train_input_fn = tf.estimator.inputs.pandas_input_fn(x = traindf[list(featcols.keys())],
                                                    y = traindf["median_house_value"] / SCALE,
                                                    num_epochs = None,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x = evaldf[list(featcols.keys())],
                                                    y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                                    num_epochs = 1, 
                                                    batch_size = len(evaldf), 
                                                    shuffle=False)


# In[10]:


# Linear Regressor
def train_and_evaluate(output_dir, num_train_steps):
  myopt = tf.train.FtrlOptimizer(learning_rate = 0.01) # note the learning rate
  estimator = tf.estimator.LinearRegressor(
                       model_dir = output_dir, 
                       feature_columns = featcols.values(),
                       optimizer = myopt)
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
  estimator = tf.contrib.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = train_input_fn,
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = eval_input_fn,
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = (100 * len(traindf)) / BATCH_SIZE) 


# In[11]:


# DNN Regressor
def train_and_evaluate(output_dir, num_train_steps):
  myopt = tf.train.FtrlOptimizer(learning_rate = 0.01) # note the learning rate
  estimator = tf.estimator.DNNRegressor( # TODO: Implement DNN Regressor model
    model_dir = output_dir,
    hidden_units = [100, 50, 20],
    feature_columns = featcols.values(),
    optimizer = myopt,
    dropout = 0.1
  )
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
  estimator = tf.contrib.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = train_input_fn,
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = eval_input_fn,
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = (100 * len(traindf)) / BATCH_SIZE) 


# In[14]:


from google.datalab.ml import TensorBoard
pid = TensorBoard().start(OUTDIR)


# In[15]:


TensorBoard().stop(pid)
